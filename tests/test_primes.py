from typing import Iterator
from torch_exid import ExtendedIterableDataset
from math import isqrt


class PrimesDataset(ExtendedIterableDataset):
    @classmethod
    def is_prime(cls, n: int) -> bool:
        if n <= 1:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False

        for i in range(3, isqrt(n) + 1, 2):
            if n % i == 0:
                return False

        return True

    def generator(self) -> Iterator[int]:
        n = 0
        while True:
            if self.is_prime(n):
                yield n

            n += 1


def test_primes():
    ds = PrimesDataset(limit=10)
    assert list(ds) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    ds = PrimesDataset(limit=10, offset=3)
    assert list(ds) == [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
