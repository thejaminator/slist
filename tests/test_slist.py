import pytest
from pytest import CaptureFixture

from slist import Slist, identity, Group
import numpy as np


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


def test_repeat_until_size_enumerate():
    with pytest.raises(ValueError):
        Slist([]).repeat_until_size_enumerate(5)  # Empty list should raise ValueError

    result = Slist([1, 2, 3]).repeat_until_size_enumerate(5)
    assert result == Slist([(0, 1), (0, 2), (0, 3), (1, 1), (1, 2)]), "should repeat with repetition count"

    result = Slist(["a"]).repeat_until_size_enumerate(3)
    assert result == Slist([(0, "a"), (1, "a"), (2, "a")]), "single element repeated with increasing count"

    # Test with exact size match
    result = Slist([1, 2]).repeat_until_size_enumerate(4)
    assert result == Slist([(0, 1), (0, 2), (1, 1), (1, 2)]), "should have exact number of elements"


def test_split_by():
    assert Slist([]).split_by(lambda x: x % 2 == 0) == (
        Slist([]),
        Slist([]),
    ), "should split an empty Slist correctly into two empty Slists"
    assert Slist([1, 2, 3, 4, 5]).split_by(lambda x: x % 2 == 0) == (
        Slist([2, 4]),
        Slist([1, 3, 5]),
    ), "should split a non-empty Slist correctly into two Slists"
    assert (
        Slist([1, 2, 3, 4, 5]).split_by(lambda x: True)
        == (
            Slist([1, 2, 3, 4, 5]),
            Slist([]),
        )
    ), "should split a non-empty Slist with an always True predicate with all elements in left Slist, and no elements on right Slist"
    assert (
        Slist([1, 2, 3, 4, 5]).split_by(lambda x: False)
        == (
            Slist([]),
            Slist([1, 2, 3, 4, 5]),
        )
    ), "should split a non-empty Slist with an always True predicate with all elements in left Slist, and no elements on right Slist"


def test_split_on():
    assert Slist([1, 2, 3, 4, 5]).split_on(lambda x: x == 3) == Slist(
        [
            Slist([1, 2]),
            Slist([4, 5]),
        ]
    )
    assert Slist(["hello", "", "world"]).split_on(lambda x: x == "") == Slist(
        [
            Slist(["hello"]),
            Slist(["world"]),
        ]
    )
    assert Slist(["hello", "world"]).split_on(lambda x: x == "") == Slist(
        [
            Slist(["hello", "world"]),
        ]
    )


def test_find_last_idx_or_raise():
    assert Slist([1, 1, 1, 1]).find_last_idx_or_raise(lambda x: x == 1) == 3


def test_zip():
    assert Slist([]).zip(Slist([])) == Slist([])
    assert Slist([1, 2, 3]).zip(Slist(["1", "2", "3"])) == Slist([(1, "1"), (2, "2"), (3, "3")])
    assert Slist([1, 2, 3]).zip(Slist(["1", "2", "3"]), Slist([True, True, True])) == Slist(
        [(1, "1", True), (2, "2", True), (3, "3", True)]
    )

    with pytest.raises(ValueError):
        Slist([1, 2, 3]).zip(Slist(["1"]))
    with pytest.raises(ValueError):
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
    async def func(x: int) -> int:
        return x * 2

    result = await Slist([1, 2, 3]).par_map_async(func)
    assert result == Slist([2, 4, 6])


@pytest.mark.asyncio
async def test_par_map_async_max_parallel():
    async def func(x: int) -> int:
        return x * 2

    result = await Slist([1, 2, 3]).par_map_async(func, max_par=1)
    assert result == Slist([2, 4, 6])


@pytest.mark.asyncio
async def test_par_map_async_max_parallel_tqdm():
    async def func(x: int) -> int:
        return x * 2

    result = await Slist([1, 2, 3]).par_map_async(func, max_par=1, tqdm=True)
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


def test_max_option():
    numbers = Slist([2, 3, 4, 5, 6, 7, 8, 9, 1])
    assert numbers.max_option() == 9
    empty = Slist([])
    assert empty.max_option() is None


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


def test_fold_left_add():
    numbers = Slist([1, 2, 3, 4, 5])
    assert numbers.sum_option() == 15
    empty = Slist([])
    assert empty.sum_option() is None


def test_group_by():
    class Animal:
        def __init__(self, name: str, age: int):
            self.name = name
            self.age = age

    animals = Slist(
        [
            Animal("cat", 1),
            Animal("cat", 2),
            Animal("dog", 1),
        ]
    )
    # group_by name, then average out the age
    result = animals.group_by(lambda animal: animal.name).map(
        lambda group: group.map_values(lambda animal: animal.map(lambda x: x.age).average_or_raise())
    )
    assert result == Slist(
        [
            ("cat", 1.5),
            ("dog", 1),
        ]
    )


def test_group_by_len():
    class Animal:
        def __init__(self, name: str, age: int):
            self.name = name
            self.age = age

    animals = Slist(
        [
            Animal("cat", 1),
            Animal("cat", 2),
            Animal("dog", 1),
        ]
    )
    # group_by name, then average out the age
    result = animals.group_by(lambda animal: animal.name).map(lambda group: group.map_values(len))
    assert result == Slist(
        [
            ("cat", 2),
            ("dog", 1),
        ]
    )


def test_group_by_map_2():
    class Animal:
        def __init__(self, name: str, age: int):
            self.name = name
            self.age = age

    animals = Slist(
        [
            Animal("cat", 1),
            Animal("cat", 2),
            Animal("dog", 1),
        ]
    )
    # group_by name, then average out the age
    group = animals.group_by(lambda animal: animal.name)
    result = group.map_2(
        lambda group_name, group_values: (group_name, group_values.map(lambda animal: animal.age).average_or_raise())
    )
    assert result == Slist(
        [
            ("cat", 1.5),
            ("dog", 1),
        ]
    )


def test_take_or_raise():
    numbers = Slist([1, 2, 3, 4, 5])
    assert numbers.take_or_raise(0) == Slist([])
    assert numbers.take_or_raise(1) == Slist([1])
    assert numbers.take_or_raise(2) == Slist([1, 2])
    assert numbers.take_or_raise(5) == Slist([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        numbers.take_or_raise(6)


def test_product():
    numbers = Slist([1, 2, 3, 4, 5])
    # cartesian product
    assert numbers.product(numbers) == Slist(
        [
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (3, 5),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 5),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
        ]
    )


def test_statistics_or_raise():
    numbers = Slist([1, 2, 3, 4, 5])
    results = numbers.statistics_or_raise()
    assert results.average == 3
    assert results.count == 5

    # convert the above to use numpy roughly equal
    assert np.isclose(results.upper_confidence_interval_95, 4.38, atol=0.01)
    assert np.isclose(results.lower_confidence_interval_95, 1.61, atol=0.01)

    empty = Slist([])
    with pytest.raises(ValueError):
        empty.statistics_or_raise()


def test_one():
    assert Slist.one(1) == Slist([1])
    assert Slist.one("test") == Slist(["test"])
    assert Slist.one([1, 2]) == Slist([[1, 2]])


def test_one_option():
    assert Slist.one_option(1) == Slist([1])
    assert Slist.one_option(None) == Slist()
    assert Slist.one_option("test") == Slist(["test"])


def test_any():
    numbers = Slist([1, 2, 3, 4, 5])
    assert numbers.any(lambda x: x > 3) is True
    assert numbers.any(lambda x: x > 5) is False
    assert Slist([]).any(lambda x: True) is False


def test_all():
    numbers = Slist([1, 2, 3, 4, 5])
    assert numbers.all(lambda x: x > 0) is True
    assert numbers.all(lambda x: x > 3) is False
    assert Slist([]).all(lambda x: False) is True


def test_max_by_ordering():
    numbers = Slist([1, 2, 3, 4, 5])
    # Find maximum by comparing if first number is less than second
    assert numbers.max_by_ordering(lambda x, y: x < y) == 5
    # Find maximum by comparing if first number is greater than second (will give minimum)
    assert numbers.max_by_ordering(lambda x, y: x > y) == 1
    assert Slist([]).max_by_ordering(lambda x, y: x < y) is None


def test_slice_with_bool():
    numbers = Slist([1, 2, 3, 4, 5])
    bools = [True, False, True, False, True]
    assert numbers.slice_with_bool(bools) == Slist([1, 3, 5])
    assert numbers.slice_with_bool([False, False, False, False, False]) == Slist([])
    assert numbers.slice_with_bool([True, True, True, True, True]) == numbers


def test_find_one_or_raise():
    numbers = Slist([1, 2, 3, 4, 5])
    assert numbers.find_one_or_raise(lambda x: x > 3) == 4
    assert numbers.find_one_or_raise(lambda x: x == 1) == 1
    with pytest.raises(RuntimeError, match="Failed to find predicate"):
        numbers.find_one_or_raise(lambda x: x > 10)
    with pytest.raises(ValueError, match="Custom error"):
        numbers.find_one_or_raise(lambda x: x > 10, ValueError("Custom error"))


def test_pairwise():
    numbers = Slist([1, 2, 3, 4, 5])
    assert numbers.pairwise() == Slist([(1, 2), (2, 3), (3, 4), (4, 5)])
    assert Slist([1]).pairwise() == Slist([])
    assert Slist([]).pairwise() == Slist([])


def test_print_length(capsys: CaptureFixture[str]):
    numbers = Slist([1, 2, 3, 4, 5])
    result = numbers.print_length()
    captured = capsys.readouterr()
    assert captured.out == "Slist Length: 5\n"
    assert result == numbers  # Should return the original list

    # Test with custom prefix
    result = numbers.print_length(prefix="Length: ")
    captured = capsys.readouterr()
    assert captured.out == "Length: 5\n"

    # Test with custom printer
    output = []
    numbers.print_length(printer=output.append)
    assert output == ["Slist Length: 5"]


def test_empty_properties():
    empty_list = Slist([])
    assert empty_list.is_empty is True
    assert empty_list.not_empty is False

    non_empty_list = Slist([1, 2, 3])
    assert non_empty_list.is_empty is False
    assert non_empty_list.not_empty is True


def test_value_counts():
    # Test with strings
    data = Slist(["apple", "banana", "apple", "cherry", "banana", "apple"])
    result = data.value_counts(key=lambda x: x)
    assert result == Slist([Group(key="apple", values=3), Group(key="banana", values=2), Group(key="cherry", values=1)])

    # Test with sort=False should preserve input order
    data = Slist(["cherry", "banana", "apple", "banana", "apple", "apple"])
    result = data.value_counts(key=lambda x: x, sort=False)
    assert result == Slist([Group(key="cherry", values=1), Group(key="banana", values=2), Group(key="apple", values=3)])

    # Test with numbers
    numbers = Slist([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    result = numbers.value_counts(key=lambda x: x)
    assert result == Slist(
        [Group(key=4, values=4), Group(key=3, values=3), Group(key=2, values=2), Group(key=1, values=1)]
    )

    # Test with custom key function
    data = Slist(["apple", "banana", "cherry", "date", "elderberry"])
    result = data.value_counts(key=lambda x: len(x))
    assert result == Slist(
        [Group(key=6, values=2), Group(key=5, values=1), Group(key=4, values=1), Group(key=10, values=1)]
    )

    # Test with empty list
    empty = Slist([])
    result = empty.value_counts(key=lambda x: x)
    assert result == Slist([])


def test_value_percentage():
    # Test with strings
    data = Slist(["apple", "banana", "apple", "cherry", "banana", "apple"])
    result = data.value_percentage(key=lambda x: x)

    assert len(result) == 3
    assert result[0].key == "apple"
    assert result[1].key == "banana"
    assert result[2].key == "cherry"
    assert pytest.approx(result[0].values) == 0.5  # 3/6
    assert pytest.approx(result[1].values) == 1 / 3  # 2/6
    assert pytest.approx(result[2].values) == 1 / 6  # 1/6

    # Test with sort=False should preserve input order
    data = Slist(["cherry", "banana", "apple", "banana", "apple", "apple"])
    result = data.value_percentage(key=lambda x: x, sort=False)

    assert len(result) == 3
    assert result[0].key == "cherry"
    assert result[1].key == "banana"
    assert result[2].key == "apple"
    assert pytest.approx(result[0].values) == 1 / 6  # 1/6
    assert pytest.approx(result[1].values) == 1 / 3  # 2/6
    assert pytest.approx(result[2].values) == 0.5  # 3/6

    # Test with numbers
    numbers = Slist([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    result = numbers.value_percentage(key=lambda x: x)

    assert len(result) == 4
    assert result[0].key == 4
    assert result[1].key == 3
    assert result[2].key == 2
    assert result[3].key == 1
    assert pytest.approx(result[0].values) == 0.4  # 4/10
    assert pytest.approx(result[1].values) == 0.3  # 3/10
    assert pytest.approx(result[2].values) == 0.2  # 2/10
    assert pytest.approx(result[3].values) == 0.1  # 1/10

    # Test with custom key function
    data = Slist(["apple", "banana", "cherry", "date", "elderberry"])
    result = data.value_percentage(key=lambda x: len(x))

    assert len(result) == 4
    assert "date" in data  # 4 chars
    assert "apple" in data  # 5 chars
    assert "banana" in data and "cherry" in data  # 6 chars
    assert "elderberry" in data  # 10 chars

    # Verify all keys are present
    keys = [group.key for group in result]
    assert 4 in keys
    assert 5 in keys
    assert 6 in keys
    assert 10 in keys

    # Verify values add up to 1.0
    total = sum(group.values for group in result)
    assert pytest.approx(total) == 1.0

    # Verify each value is 0.2 (1/5) except for length 6 which should be 0.4 (2/5)
    for group in result:
        if group.key == 6:
            assert pytest.approx(group.values) == 0.4  # 2/5
        else:
            assert pytest.approx(group.values) == 0.2  # 1/5

    # Test with empty list
    empty = Slist([])
    result = empty.value_percentage(key=lambda x: x)
    assert result == Slist([])
