from typing import Iterator
from torch_exid import ExtendedIterableDataset


class EvensDataset(ExtendedIterableDataset):
    def generator(self) -> Iterator[int]:
        n = 0
        while True:
            if n % 2 != 0:
                self.skip_next()

            yield n
            n += 1


def test_skip():
    ds = EvensDataset(limit=5)
    assert list(ds) == [0, 2, 4, 6, 8]
    assert list(ds) == [0, 2, 4, 6, 8]

    ds = EvensDataset(limit=5, offset=3)
    assert list(ds) == [6, 8, 10, 12, 14]
    assert list(ds) == [6, 8, 10, 12, 14]

    ds = EvensDataset(limit=5, shuffle_buffer=3, shuffle_seed=42)
    assert list(ds) == [2, 0, 4, 8, 6]
    assert list(ds) == [2, 0, 4, 8, 6]
