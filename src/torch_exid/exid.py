from random import Random
from typing import Any, Iterator, List, Callable, Optional
from torch.utils.data import IterableDataset


class ExtendedIterableDataset(IterableDataset):
    """
    This class extends IterableDataset by allowing data shuffling, applying transformations, and limiting the amount of data.

    Parameters
    ----------
    shuffle_buffer : int
        The size of the shuffle buffer. If it's greater than 1, data is buffered and shuffled before being returned.
    shuffle_seed : Optional[int]
        The seed for the random number generator used for shuffling.
    offset : int
        The number of first data points to skip.
    limit : int
        The maximum number of data points to return. If it's negative, all data is returned.
    transforms : List[Callable[[Any], Any]]
        A list of transformations to apply to the data.
    transforms_required : bool
        If it's true and transforms is empty, an exception is raised.
    *args
        Additional arguments for the IterableDataset.
    **kwargs
        Additional keyword arguments for the IterableDataset.
    """

    def __init__(
        self,
        shuffle_buffer: int = 1,
        shuffle_seed: Optional[int] = None,
        offset: int = 0,
        limit: int = -1,
        transforms: Optional[List[Callable[[Any], Any]]] = None,
        transforms_required: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        """
        Some datasets may use auxiliary classes as singletons to generate data.
        For example, a reusable chess.Board instance.
        
        In order to avoid bugs with its state,
        use the "transforms_required" flag to warn the user
        that data should be extracted from this instance.
        """
        if transforms_required and not transforms:
            raise ValueError(
                f"ExtendedIterableDataset requires transforms in order to avoid bugs. Please read the comment above"
            )

        self.shuffle_buffer = shuffle_buffer
        self.shuffle_seed = (
            shuffle_seed if shuffle_seed is not None else Random().randint(0, 999331)
        )
        self.buffer = []

        self.offset = offset
        self.limit = limit
        self.counter = 0

        self.transforms = transforms if transforms is not None else []

    def generator(self) -> Iterator[Any]:
        """
        Should yield items. This method should be implemented in subclasses.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError("please implement generator method")

    @property
    def limit_allows_one_more(self) -> bool:
        """
        Checks if the dataset limit allows to yield one more data item.

        Returns
        -------
        bool
            True if the dataset limit allows to yield one more data item, False otherwise.
        """

        if self.limit < 0:
            return True

        return self.counter < self.limit + self.offset

    def transform_x(self, x: Any) -> Any:
        """
        Applies transformations to the data.

        Parameters
        ----------
        x : Any
            The data point to be transformed.

        Returns
        -------
        Any
            The transformed data point.
        """
        for transform in self.transforms:
            x = transform(x)

        return x

    def flush_buffer(self) -> Iterator[Any]:
        """
        Shuffles and yields the buffer content, then resets the buffer.

        Returns
        -------
        Iterator[Any]
            An iterator over the shuffled buffer content.
        """

        Random(self.shuffle_seed).shuffle(self.buffer)

        for x in self.buffer:
            yield x

        self.buffer = []

    def generator_with_conditions(self) -> Iterator[Any]:
        """
        Generates items taking into account the dataset limit and offset.
        Applies transformations to each generated item.

        Returns
        -------
        Iterator[Any]
            An iterator over the generated and transformed items.

        Raises
        ------
        StopIteration
            If the dataset limit does not allow to yield more items.
        """

        gen = self.generator()

        while True:
            try:
                if not self.limit_allows_one_more:
                    raise StopIteration

                x = next(gen)

                if self.counter < self.offset:
                    self.counter += 1
                    continue

                yield self.transform_x(x)

                self.counter += 1

            except StopIteration:
                break

    def generator_with_buffer(self) -> Iterator[Any]:
        """
        Generates items with a shuffle buffer.
        If the buffer size is greater than 1, items are buffered and shuffled before being yielded.
        If the buffer size is 1 or less, items are yielded as they are generated.

        Returns
        -------
        Iterator[Any]
            An iterator over the generated, possibly shuffled items.
        """

        if self.shuffle_buffer > 1:
            for x in self.generator_with_conditions():
                self.buffer.append(x)

                if len(self.buffer) >= self.shuffle_buffer:
                    yield from self.flush_buffer()

            if len(self.buffer) > 0:
                yield from self.flush_buffer()

        else:
            for x in self.generator_with_conditions():
                yield x

        # Reset the counter and the buffer
        self.counter = 0
        self.buffer = []

    def __iter__(self) -> Iterator[Any]:
        """
        Generates items with a shuffle buffer.
        If the buffer size is greater than 1, items are buffered and shuffled before being yielded.
        If the buffer size is 1 or less, items are yielded as they are generated.

        Returns
        -------
        Iterator[Any]
            An iterator over the generated, possibly shuffled items.
        """
        return self.generator_with_buffer()
