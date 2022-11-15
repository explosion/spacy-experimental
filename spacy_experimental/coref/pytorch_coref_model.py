from typing import List, Tuple

import torch
from torch import nn

EPSILON = 1e-7


class CorefClusterer(nn.Module):
    """
    Combines all coref modules together to find coreferent token pairs.
    Submodules (in the order of their usage in the pipeline):
        - rough_scorer (RoughScorer) that prunes candidate pairs
        - pairwise (DistancePairwiseEncoder) that computes pairwise features
        - ana_scorer (AnaphoricityScorer) produces the final scores
    """

    def __init__(
        self,
        dim: int,
        dist_emb_size: int,
        hidden_size: int,
        n_layers: int,
        dropout: float,
        rough_k: int,
        batch_size: int,
    ):
        super().__init__()
        """
        dim: Size of the input features.
        dist_emb_size: Size of the distance embeddings.
        hidden_size: Size of the coreference candidate embeddings.
        n_layers: Numbers of layers in the AnaphoricityScorer.
        dropout: Dropout probability to apply across all modules.
        rough_k: Number of candidates the RoughScorer returns.
        batch_size: Internal batch-size for the more expensive scorer.
        """
        self.dropout = torch.nn.Dropout(dropout)
        self.batch_size = batch_size
        self.pairwise = DistancePairwiseEncoder(dist_emb_size, dropout)

        pair_emb = dim * 3 + self.pairwise.shape
        self.ana_scorer = AnaphoricityScorer(pair_emb, hidden_size, n_layers, dropout)
        self.lstm = torch.nn.LSTM(
            input_size=dim,
            hidden_size=dim,
            batch_first=True,
        )

        self.rough_scorer = RoughScorer(dim, dropout, rough_k)

    def forward(self, word_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        1. LSTM encodes the incoming word_features.
        2. The RoughScorer scores and prunes the candidates.
        3. The DistancePairwiseEncoder embeds the distances between pairs.
        4. The AnaphoricityScorer scores all pairs in mini-batches.

        word_features: torch.Tensor containing word encodings

        returns:
            coref_scores: n_words x rough_k floats.
            top_indices: n_words x rough_k integers.
        """
        self.lstm.flatten_parameters()  # XXX without this there's a warning
        word_features = torch.unsqueeze(word_features, dim=0)
        words, _ = self.lstm(word_features)
        words = words.squeeze()
        # words: n_words x dim
        words = self.dropout(words)
        # Obtain bilinear scores and leave only top-k antecedents for each word
        # top_rough_scores: (n_words x rough_k)
        # top_indices: (n_words x rough_k)
        top_rough_scores, top_indices = self.rough_scorer(words)
        # Get pairwise features
        # (n_words x rough_k x n_pairwise_features)
        pairwise = self.pairwise(top_indices)
        batch_size = self.batch_size
        a_scores_lst: List[torch.Tensor] = []

        for i in range(0, len(words), batch_size):
            pairwise_batch = pairwise[i : i + batch_size]
            words_batch = words[i : i + batch_size]
            top_indices_batch = top_indices[i : i + batch_size]
            top_rough_scores_batch = top_rough_scores[i : i + batch_size]

            # a_scores_batch    [batch_size, n_ants]
            a_scores_batch = self.ana_scorer(
                all_mentions=words,
                mentions_batch=words_batch,
                pairwise_batch=pairwise_batch,
                top_indices_batch=top_indices_batch,
                top_rough_scores_batch=top_rough_scores_batch,
            )
            a_scores_lst.append(a_scores_batch)

        coref_scores = torch.cat(a_scores_lst, dim=0)
        return coref_scores, top_indices


# Note this function is kept here to keep a torch dep out of coref_util.
def add_dummy(tensor: torch.Tensor, eps: bool = False):
    """Prepends zeros (or a very small value if eps is True)
    to the first (not zeroth) dimension of tensor.
    """
    kwargs = dict(device=tensor.device, dtype=tensor.dtype)
    shape: List[int] = list(tensor.shape)
    shape[1] = 1
    if not eps:
        dummy = torch.zeros(shape, **kwargs)  # type: ignore
    else:
        dummy = torch.full(shape, EPSILON, **kwargs)  # type: ignore
    output = torch.cat((dummy, tensor), dim=1)
    return output


class AnaphoricityScorer(nn.Module):
    """Calculates anaphoricity scores by passing the inputs into a FFNN"""

    def __init__(self, in_features: int, hidden_size, depth, dropout):
        super().__init__()
        hidden_size = hidden_size
        if not depth:
            hidden_size = in_features
        layers = []
        for i in range(depth):
            layers.extend(
                [
                    torch.nn.Linear(hidden_size if i else in_features, hidden_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Dropout(dropout),
                ]
            )
        self.hidden = torch.nn.Sequential(*layers)
        self.out = torch.nn.Linear(hidden_size, out_features=1)

    def forward(
        self,
        *,  # type: ignore  # pylint: disable=arguments-differ  #35566 in pytorch
        all_mentions: torch.Tensor,
        mentions_batch: torch.Tensor,
        pairwise_batch: torch.Tensor,
        top_indices_batch: torch.Tensor,
        top_rough_scores_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Builds a pairwise matrix, scores the pairs and returns the scores.

        Args:
            all_mentions (torch.Tensor): [n_mentions, mention_emb]
            mentions_batch (torch.Tensor): [batch_size, mention_emb]
            pairwise_batch (torch.Tensor): [batch_size, n_ants, pairwise_emb]
            top_indices_batch (torch.Tensor): [batch_size, n_ants]
            top_rough_scores_batch (torch.Tensor): [batch_size, n_ants]

        Returns:
            torch.Tensor [batch_size, n_ants + 1]
                anaphoricity scores for the pairs + a dummy column
        """
        # [batch_size, n_ants, pair_emb]
        pair_matrix = self._get_pair_matrix(
            all_mentions, mentions_batch, pairwise_batch, top_indices_batch
        )

        # [batch_size, n_ants]
        scores = top_rough_scores_batch + self._ffnn(pair_matrix)
        scores = add_dummy(scores, eps=True)

        return scores

    def _ffnn(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape (batch_size x rough_k x n_features
        returns: tensor of shape (batch_size x antecedent_limit)
        """
        x = self.out(self.hidden(x))
        return x.squeeze(2)

    @staticmethod
    def _get_pair_matrix(
        all_mentions: torch.Tensor,
        mentions_batch: torch.Tensor,
        pairwise_batch: torch.Tensor,
        top_indices_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Builds the matrix used as input for AnaphoricityScorer.

        all_mentions: (n_mentions x mention_emb),
            all the valid mentions of the document,
            can be on a different device
        mentions_batch: (batch_size x mention_emb),
            the mentions of the current batch.
        pairwise_batch: (batch_size x rough_k x pairwise_emb),
            pairwise distance features of the current batch.
        top_indices_batch: (batch_size x n_ants),
            indices of antecedents of each mention

        Returns:
            out: pairwise features (batch_size x n_ants x pair_emb)
        """
        emb_size = mentions_batch.shape[1]
        n_ants = pairwise_batch.shape[1]

        a_mentions = mentions_batch.unsqueeze(1).expand(-1, n_ants, emb_size)
        b_mentions = all_mentions[top_indices_batch]
        similarity = a_mentions * b_mentions

        out = torch.cat((a_mentions, b_mentions, similarity, pairwise_batch), dim=2)
        return out


class RoughScorer(nn.Module):
    """
    Cheaper module that gives a rough estimate of the anaphoricity of two
    candidates, only top scoring candidates are considered on later
    steps to reduce computational cost.
    """

    def __init__(self, features: int, dropout: float, antecedent_limit: int):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.bilinear = torch.nn.Linear(features, features)
        self.k = antecedent_limit

    def forward(
        self,  # type: ignore
        mentions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns rough anaphoricity scores for candidates, which consist of
        the bilinear output of the current model summed with mention scores.
        """
        # [n_mentions, n_mentions]
        pair_mask = torch.arange(mentions.shape[0])
        pair_mask = pair_mask.unsqueeze(1) - pair_mask.unsqueeze(0)
        pair_mask = torch.log((pair_mask > 0).to(torch.float))
        pair_mask = pair_mask.to(mentions.device)
        bilinear_scores = self.dropout(self.bilinear(mentions)).mm(mentions.T)
        rough_scores = pair_mask + bilinear_scores
        top_scores, indices = torch.topk(
            rough_scores, k=min(self.k, len(rough_scores)), dim=1, sorted=False
        )

        return top_scores, indices


class DistancePairwiseEncoder(nn.Module):
    def __init__(self, distance_embedding_size, dropout):
        """
        Takes the top_indices indicating, which is a ranked
        list for each word and its most likely corresponding
        anaphora candidates. For each of these pairs it looks
        up a distance embedding from a table, where the distance
        corresponds to the log-distance.

        distance_embedding_size: int,
            Dimensionality of the distance-embeddings table.
        dropout: float,
            Dropout probability.
        """
        super().__init__()
        emb_size = distance_embedding_size
        self.distance_emb = torch.nn.Embedding(9, emb_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.shape = emb_size

    def forward(self, top_indices: torch.Tensor) -> torch.Tensor:
        word_ids = torch.arange(0, top_indices.size(0), device=top_indices.device)
        distance = (word_ids.unsqueeze(1) - word_ids[top_indices]).clamp_min_(min=1)
        log_distance = distance.to(torch.float).log2().floor_()
        log_distance = log_distance.clamp_max_(max=6).to(torch.long)
        distance = torch.where(distance < 5, distance - 1, log_distance + 2)
        distance = distance.to(top_indices.device)
        distance = self.distance_emb(distance)
        return self.dropout(distance)
