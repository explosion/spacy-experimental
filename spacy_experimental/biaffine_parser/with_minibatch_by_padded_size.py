from typing import Callable, Generic, List, Optional, Sized, Tuple, TypeVar, cast
from dataclasses import dataclass
from spacy.training.batchers import minibatch_by_padded_size
from thinc.api import Model


OutT = TypeVar("OutT")
SizedInT = TypeVar("SizedInT", bound=Sized)


@dataclass
class ItemIndex(Generic[SizedInT]):
    value: SizedInT
    idx: int

    def __len__(self):
        return len(self.value)


def with_minibatch_by_padded_size(
    inner: Model[List[SizedInT], List[OutT]], size: int, buffer: int = 256
) -> Model[List[SizedInT], List[OutT]]:
    """Batch the inputs sorted by length and with a maximum number of
    padded batch items."""
    return Model(
        "with_minibatch_by_padded_size",
        with_minibatch_by_padded_size_forward,
        init=with_minibatch_by_padded_size_init,
        attrs={"buffer": buffer, "size": size},
        layers=[inner],
    )


def with_minibatch_by_padded_size_init(
    model: Model[List[SizedInT], List[OutT]], X: Optional[SizedInT] = None, Y=None
) -> None:
    # Pass X through as-is. Downstream models don't need the batching
    # for proper initialization.
    model.layers[0].initialize(X=X, Y=Y)


def with_minibatch_by_padded_size_forward(
    model: Model[List[SizedInT], List[OutT]],
    X: List[SizedInT],
    is_train: bool,
) -> Tuple[List[OutT], Callable[[List[OutT]], List[SizedInT]]]:
    inner: Model[List[SizedInT], List[OutT]] = model.layers[0]
    buffer: int = model.attrs["buffer"]
    size: int = model.attrs["size"]

    batched = list(
        minibatch_by_padded_size(
            [ItemIndex(idx=idx, value=item) for idx, item in enumerate(X)],
            size,
            buffer=buffer,
        )
    )

    backprops = []
    Y: List[Optional[OutT]] = [None] * len(X)
    for batch in batched:
        X_batch = [split.value for split in batch]
        Y_batch, backprop_batch = inner(X_batch, is_train)
        backprops.append(backprop_batch)

        # Place in outputs.
        offsets_batch = [split.idx for split in batch]
        for split_offset, Y_split in zip(offsets_batch, Y_batch):
            Y[split_offset] = Y_split

    assert not any(y is None for y in Y)

    def backprop(dY: List[OutT]) -> List[SizedInT]:
        dX: List[Optional[SizedInT]] = [None] * len(X)
        for idx, batch in enumerate(batched):
            dY_batch = [dY[split.idx] for split in batch]
            for split, dX_split in zip(batch, backprops[idx](dY_batch)):
                dX[split.idx] = dX_split

        assert not any(dx is None for dx in dX)

        return cast(List[SizedInT], dX)

    return cast(List[OutT], Y), backprop
