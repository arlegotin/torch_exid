# Extended Iterable Dataset for PyTorch
An extension of PyTorch [IterableDataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset), this package introduces functionalities for shuffling, limiting, and offsetting data.

## Installation

Directly from [PyPI](https://pypi.org/project/torch_exid/):
```bash
pip install torch-exid
```

Or using [Poetry](https://python-poetry.org/):
```bash
poetry add torch-exid
```

## Usage
Begin by subclassing `ExtendedIterableDataset` and implement the `generator` method to yield items.

Here's a simple example using an `IntegersDataset`:
```python
from torch_exid import ExtendedIterableDataset

class IntegersDataset(ExtendedIterableDataset):
    def generator(self) -> Iterator[int]:
        n = 0
        while True:
            yield n
            n += 1

# Will print out integers 0, 1, ..., 9:
for n in IntegersDataset(limit=10):
    print(n)
```

## Constructor Parameters
`ExtendedIterableDataset` introduces several parameters to provide additional control:

### limit: int
Sets the maximum number of data points to return. If negative, all data points are returned. Default is `-1` (return all data).
```python
# Will print out "0, 1, 2"
for n in IntegersDataset(limit=3)
    print(n)
```

### offset: int
Determines the number of initial data points to skip. Default is `0`.
```python
# Will print out "2, 3, 4"
for n in IntegersDataset(limit=3, offset=2)
    print(n)
```

### shuffle_buffer: int
This specifies the buffer size for shuffling. If greater than `1`, data is buffered and shuffled prior to being returned. If set to `1` (default), no shuffling occurs.

```python
# Will print out "0, 1, 3, 2" for the first time...
for n in IntegersDataset(limit=4, shuffle_buffer=2)
    print(n)

# ...and 1, 0, 2, 3 second time
for n in IntegersDataset(limit=4, shuffle_buffer=2)
    print(n)
```

### shuffle_seed: int
Defines the seed for the random number generator used in shuffling. If not provided, a random seed is used:

```python
# Will print out "1, 0, 3, 2" both times:
for n in IntegersDataset(limit=4, shuffle_buffer=2, shuffle_seed=42)
    print(n)

for n in IntegersDataset(limit=4, shuffle_buffer=2, shuffle_seed=42)
    print(n)
```

### transforms: List[Callable[[Any], Any]]
A list of transformations to apply to the data. Default is an empty list.
```python
ds = IntegersDataset(
    limit=3,
    transforms=[
        lambda n: n + 1,
        lambda n: n ** 2,
    ],
)

# Will print out "1, 4, 9"
for n in ds:
    print(n)
```

In addition to the above, any arguments or keyword arguments for the [IterableDataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) superclass can also be passed.

## Contributing
Contributions are greatly appreciated! Improvement can be made by submitting issues, proposing new features, or submitting pull requests with bug fixes or new functionalities.

### Getting started with contributing
Here are the steps to get started with development:
```bash
# Clone the repository:
git clone https://github.com/arlegotin/torch_exid.git
cd torch_exid

# Install the project and its dependencies using Poetry:
poetry install

# Spawn a shell within the virtual environment:
poetry shell

# Run tests to ensure everything is working correctly:
pytest tests/
```

Please ensure all changes are accompanied by relevant unit tests, and that all tests pass before submitting a pull request. This helps maintain the quality and reliability of the project.
