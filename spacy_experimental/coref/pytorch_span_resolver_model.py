import torch
from torch import nn


class SpanResolverModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dist_emb_size: int,
        conv_channels: int,
        window_size: int,
        max_distance: int,
    ):
        super().__init__()
        if max_distance % 2 != 0:
            raise ValueError("max_distance has to be an even number")
        # input size = single token size
        # 64 = probably distance emb size
        self.ffnn = torch.nn.Sequential(
            torch.nn.Linear(input_size * 2 + dist_emb_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            # this use of dist_emb_size looks wrong but it was 64...?
            torch.nn.Linear(256, dist_emb_size),
        )
        kernel_size = window_size * 2 + 1
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(dist_emb_size, conv_channels, kernel_size, 1, 1),
            torch.nn.Conv1d(conv_channels, 2, kernel_size, 1, 1),
        )
        self.max_distance = max_distance
        # handle distances between +-(max_distance - 2 / 2)
        self.emb = torch.nn.Embedding(max_distance, dist_emb_size)

    def forward(
        self,
        sent_id,
        words: torch.Tensor,
        heads_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates span start/end scores of words for each span
        for each head.

        sent_id: Sentence id of each word.
        words: features for each word in the document.
        heads_ids: word indices of span heads

        Returns:
            torch.Tensor: span start/end scores, (n_heads x n_words x 2)
        """

        # If we don't receive heads, return empty
        device = heads_ids.device
        if heads_ids.nelement() == 0:
            return torch.empty(size=(0,))
        # Obtain distance embedding indices, [n_heads, n_words]
        relative_positions = heads_ids.unsqueeze(1) - torch.arange(
            words.shape[0], device=device
        ).unsqueeze(0)
        md = self.max_distance
        # make all valid distances positive
        emb_ids = relative_positions + (md - 2) // 2
        # "too_far"
        emb_ids[(emb_ids < 0) + (emb_ids > md - 2)] = md - 1
        # Obtain "same sentence" boolean mask: (n_heads x n_words)
        heads_ids = heads_ids.long()
        same_sent = sent_id[heads_ids].unsqueeze(1) == sent_id.unsqueeze(0)
        # To save memory, only pass candidates from one sentence for each head
        # pair_matrix contains concatenated span_head_emb + candidate_emb + distance_emb
        # for each candidate among the words in the same sentence as span_head
        # (n_heads x input_size * 2 x distance_emb_size)
        rows, cols = same_sent.nonzero(as_tuple=True)
        pair_matrix = torch.cat(
            (
                words[heads_ids[rows]],
                words[cols],
                self.emb(emb_ids[rows, cols]),
            ),
            dim=1,
        )
        lengths = same_sent.sum(dim=1)
        padding_mask = torch.arange(0, lengths.max().item(), device=device).unsqueeze(0)
        # (n_heads x max_sent_len)
        padding_mask = padding_mask < lengths.unsqueeze(1)
        # (n_heads x max_sent_len x input_size * 2 + distance_emb_size)
        # This is necessary to allow the convolution layer to look at several
        # word scores
        padded_pairs = torch.zeros(
            *padding_mask.shape, pair_matrix.shape[-1], device=device
        )
        padded_pairs[padding_mask] = pair_matrix
        res = self.ffnn(padded_pairs)  # (n_heads x n_candidates x last_layer_output)
        res = self.conv(res.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # (n_heads x n_candidates, 2)

        scores = torch.full(
            (heads_ids.shape[0], words.shape[0], 2), float("-inf"), device=device
        )
        scores[rows, cols] = res[padding_mask]
        # Make sure that start <= head <= end during inference
        if not self.training:
            valid_starts = torch.log((relative_positions >= 0).to(torch.float))
            valid_ends = torch.log((relative_positions <= 0).to(torch.float))
            valid_positions = torch.stack((valid_starts, valid_ends), dim=2)
            return scores + valid_positions
        return scores
