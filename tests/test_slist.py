import pytest

from slist import Slist, identity


def test_split_proportion():
    test_list = Slist([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    left, right = test_list.split_proportion(0.5)
    assert left == Slist([1, 2, 3, 4, 5])
    assert right == Slist([6, 7, 8, 9, 10])

    left, right = test_list.split_proportion(0.25)
    assert left == Slist([1, 2, 3])
    assert right == Slist([4, 5, 6, 7, 8, 9, 10])

    left, right = test_list.split_proportion(0.75)
    assert left == Slist([1, 2, 3, 4, 5, 6, 7, 8])
    assert right == Slist([9, 10])


def test_repeat_until_size():
    assert Slist([]).repeat_until_size(5) is None, "should give back empty list"
    assert Slist([1, 2, 3]).repeat_until_size(5) == Slist([1, 2, 3, 1, 2]), "should repeat 1 and 2"
    assert Slist([1, 2, 3, 4, 5, 6]).repeat_until_size(5) == Slist([1, 2, 3, 4, 5]), "should be truncated"


def test_split_by():
    assert Slist([]).split_by(lambda x: x % 2 == 0) == (
        Slist([]),
        Slist([]),
    ), "should split an empty Slist correctly into two empty Slists"
    assert Slist([1, 2, 3, 4, 5]).split_by(lambda x: x % 2 == 0) == (
        Slist([2, 4]),
        Slist([1, 3, 5]),
    ), "should split a non-empty Slist correctly into two Slists"
    assert Slist([1, 2, 3, 4, 5]).split_by(lambda x: True) == (
        Slist([1, 2, 3, 4, 5]),
        Slist([]),
    ), "should split a non-empty Slist with an always True predicate with all elements in left Slist, and no elements on right Slist"
    assert Slist([1, 2, 3, 4, 5]).split_by(lambda x: False) == (
        Slist([]),
        Slist([1, 2, 3, 4, 5]),
    ), "should split a non-empty Slist with an always True predicate with all elements in left Slist, and no elements on right Slist"


def test_find_last_idx_or_raise():
    assert Slist([1, 1, 1, 1]).find_last_idx_or_raise(lambda x: x == 1) == 3


def test_zip():
    assert Slist([]).zip(Slist([])) == Slist([])
    assert Slist([1, 2, 3]).zip(Slist(["1", "2", "3"])) == Slist([(1, "1"), (2, "2"), (3, "3")])
    assert Slist([1, 2, 3]).zip(Slist(["1", "2", "3"]), Slist([True, True, True])) == Slist(
        [(1, "1", True), (2, "2", True), (3, "3", True)]
    )

    with pytest.raises(TypeError):
        Slist([1, 2, 3]).zip(Slist(["1"]))
    with pytest.raises(TypeError):
        Slist([1, 2, 3]).zip(Slist(["1", "2", "3"]), Slist(["1"]))


def test_take_until_inclusive():
    assert Slist([]).take_until_inclusive(lambda x: x == 1) == Slist([])
    assert Slist([1, 2, 3]).take_until_inclusive(lambda x: x == 1) == Slist([1])
    assert Slist([1, 2, 3]).take_until_inclusive(lambda x: x == 2) == Slist([1, 2])
    assert Slist([1, 2, 3]).take_until_inclusive(lambda x: x == 3) == Slist([1, 2, 3])
    assert Slist([1, 2, 3]).take_until_inclusive(lambda x: x == 4) == Slist([1, 2, 3])
    assert Slist([1, 2, 3]).take_until_inclusive(lambda x: x == 5) == Slist([1, 2, 3])


@pytest.mark.asyncio
async def test_par_map_async():
    async def func(x):
        return x * 2

    result = await Slist([1, 2, 3]).par_map_async(func)
    assert result == Slist([2, 4, 6])


def test_grouped():
    test_list = Slist([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert test_list.grouped(2) == Slist([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    test_list_2 = Slist([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    assert test_list_2.grouped(2) == Slist([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11]])


def test_window():
    test_list = Slist([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert test_list.window(size=1) == Slist([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    assert test_list.window(size=2) == Slist([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10]])


def test_window_empty_list():
    test_list = Slist([])
    assert test_list.window(size=1) == Slist()
    assert test_list.window(size=2) == Slist()


def test_window_too_small_list():
    test_list = Slist([1])
    assert test_list.window(size=2) == Slist()


def test_median():
    numbers = Slist([2, 3, 4, 5, 6, 7, 8, 9, 1])
    assert numbers.median_by(identity) == 5


def test_percentile():
    numbers = Slist([2, 3, 4, 5, 6, 7, 8, 9, 1])
    assert numbers.percentile_by(identity, 0.5) == 5
    assert numbers.percentile_by(identity, 0.25) == 3
    assert numbers.percentile_by(identity, 0.75) == 7


def test_max_by():
    numbers = Slist([2, 3, 4, 5, 6, 7, 8, 9, 1])
    assert numbers.max_by(identity) == 9
    empty = Slist([])
    assert empty.max_by(identity) is None


def test_min_by():
    numbers = Slist([2, 3, 4, 5, 6, 7, 8, 9, 1])
    assert numbers.min_by(identity) == 1
    empty = Slist([])
    assert empty.min_by(identity) is None


def test_mode_option():
    numbers = Slist([1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1])
    assert numbers.mode_option == 1
    empty = Slist([])
    assert empty.mode_option is None
