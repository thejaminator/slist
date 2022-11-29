from typing import List

import pytest

from slist import Slist

@pytest.mark.skip
def test_flatten_list():
    nested: Slist[List[int]] = Slist([[10]])
    test: Slist[int] = nested.flatten_list()  # ok
    nested_slist: Slist[Slist[int]] = Slist([Slist([10])])
    test_slist: Slist[int] = nested_slist.flatten_list()  # ok
    Slist(
        [1]
    ).flatten_list()  # Should be  error: Invalid self argument "Slist[int]" to attribute function "flatten_list" with type "Callable[[Sequence[Sequence[B]]], Slist[B]]"
