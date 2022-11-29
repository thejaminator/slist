# Slist

A spruced up version of the built-in python list.

More post fixed methods for lovely chaining!

All these methods return a new list. They do not mutate the original list.

Leverage the latest mypy features to spot errors during coding.


[![pypi](https://img.shields.io/pypi/v/slist.svg)](https://pypi.org/project/slist)
[![python](https://img.shields.io/pypi/pyversions/slist.svg)](https://pypi.org/project/slist)
[![Build Status](https://github.com/thejaminator/slist/actions/workflows/dev.yml/badge.svg)](https://github.com/thejaminator/slist/actions/workflows/dev.yml)

```
pip install slist
```

Immutable list replacement for python. With postfix methods for easy functional programming.


* GitHub: <https://github.com/thejaminator/slist>


## Quick Start
With mypy installed, easily spot errors when you call the wrong methods on your sequence.

```python
from slist import Slist

many_strings = Slist(["Lucy, Damion, Jon"])  # Slist[str]
many_strings.sum()  # Mypy errors with 'Invalid self argument'. You can't sum a sequence of strings!

many_nums = Slist([1, 1.2])
assert many_nums.sum() == 2.2  # ok!

class CannotSortMe:
    def __init__(self, value: int):
        self.value: int = value

stuff = Slist([CannotSortMe(value=1), CannotSortMe(value=1)])
stuff.sort_by(lambda x: x)  # Mypy errors with 'Cannot be "CannotSortMe"'. There isn't a way to sort by the class itself
stuff.sort_by(lambda x: x.value)  # ok! You can sort by the value

Slist([{"i am a dict": "value"}]).distinct_by(
    lambda x: x
)  # Mypy errors with 'Cannot be Dict[str, str]. You can't hash a dict itself
```

Slist provides methods that you can chain easily for easier data processing.
```python
from slist import Slist

test = Slist([-1, 0, 1]).map(
    lambda x: x if x >= 0 else None
).flatten_option()  # Mypy infers slist[int] correctly
```
