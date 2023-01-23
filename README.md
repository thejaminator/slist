# Slist
This is a drop in replacement for the built-in mutable python list

But with more post-fixed methods for chaining in a typesafe manner!!

Leverage the latest mypy features to spot errors during coding.

All these methods return a new list. They do not mutate the original list.

Not able to convince your colleagues to use immutable functional data structures? I understand.   
This library lets you still have the benefits of typesafe chaining methods while letting your colleagues have their mutable lists!





[![pypi](https://img.shields.io/pypi/v/slist.svg)](https://pypi.org/project/slist)
[![python](https://img.shields.io/pypi/pyversions/slist.svg)](https://pypi.org/project/slist)
[![Build Status](https://github.com/thejaminator/slist/actions/workflows/dev.yml/badge.svg)](https://github.com/thejaminator/slist/actions/workflows/dev.yml)

```
pip install slist
```


* GitHub: <https://github.com/thejaminator/slist>


## Quick Start
Easily spot errors when you call the wrong methods on your sequence with mypy.

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

Slist provides methods to easily flatten and infer the types of your data.
```python
from slist import Slist, List
from typing import Optional

test_optional: Slist[Optional[int]] = Slist([-1, 0, 1]).map(
    lambda x: x if x >= 0 else None
)
# Mypy infers slist[int] correctly
test_flattened: Slist[int] = test_optional.flatten_option()


test_nested: Slist[List[str]] = Slist([["bob"], ["dylan", "chan"]])
# Mypy infers slist[str] correctly
test_flattened_str: Slist[str] = test_nested.flatten_list()
```

There are plenty more methods to explore!
```python
from slist import Slist

result = (
    Slist([1, 2, 3])
    .repeat_until_size_or_raise(20)
    .grouped(2)
    .map(lambda inner_list: inner_list[0] + inner_list[1] if inner_list.length == 2 else inner_list[0])
    .flatten_option()
    .distinct_by(lambda x: x)
    .map(str)
    .reversed()
    .mk_string(sep=",")
)
assert result == "5,4,3"
```
