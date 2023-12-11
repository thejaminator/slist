from __future__ import annotations

import asyncio
import concurrent.futures
import random
import re
import statistics
import sys
import typing
from collections import OrderedDict

from functools import reduce
from itertools import tee
from typing import (
    Generic,
    TypeVar,
    Hashable,
    Protocol,
    Callable,
    Optional,
    List,
    Union,
    Sequence,
    overload,
    Any,
    Tuple,
)

# Needed for https://github.com/python/typing_extensions/issues/7
from typing_extensions import NamedTuple

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
D = TypeVar("D")
E = TypeVar("E")

CanCompare = TypeVar("CanCompare", bound="Comparable")
CanHash = TypeVar("CanHash", bound=Hashable)

identity = lambda x: x


class Group(NamedTuple, Generic[A, B]):
    """This is a NamedTuple so that you can easily access the key and values"""

    key: A
    values: B

    def map_key(self, func: Callable[[A], C]) -> Group[C, B]:
        return Group(func(self.key), self.values)

    def map_values(self, func: Callable[[B], C]) -> Group[A, C]:
        return Group(self.key, func(self.values))


class Addable(Protocol):
    def __add__(self: A, other: A, /) -> A:
        ...


CanAdd = TypeVar("CanAdd", bound=Addable)


class Comparable(Protocol):
    def __lt__(self: CanCompare, other: CanCompare, /) -> bool:
        ...

    def __gt__(self: CanCompare, other: CanCompare, /) -> bool:
        ...

    def __le__(self: CanCompare, other: CanCompare, /) -> bool:
        ...

    def __ge__(self: CanCompare, other: CanCompare, /) -> bool:
        ...


class Slist(List[A]):
    @staticmethod
    def one(element: A) -> Slist[A]:
        return Slist([element])

    @staticmethod
    def one_option(element: Optional[A]) -> Slist[A]:
        """Returns a list with one element, or an empty slist if the element is None
        Equal to Slist.one(element).flatten_option()"""
        return Slist([element]) if element is not None else Slist()

    def any(self, predicate: Callable[[A], bool]) -> bool:
        for x in self:
            if predicate(x):
                return True
        return False

    def all(self, predicate: Callable[[A], bool]) -> bool:
        for x in self:
            if not predicate(x):
                return False
        return True

    def filter(self, predicate: Callable[[A], bool]) -> Slist[A]:
        return Slist(filter(predicate, self))

    def map(self, func: Callable[[A], B]) -> Slist[B]:
        return Slist(func(item) for item in self)

    def map_2(self: Sequence[Tuple[B, C]], func: Callable[[B, C], D]) -> Slist[D]:
        return Slist(func(b, c) for b, c in self)

    def map_enumerate(self, func: Callable[[int, A], B]) -> Slist[B]:
        return Slist(func(idx, item) for idx, item in enumerate(self))

    def flatten_option(self: Sequence[Optional[B]]) -> Slist[B]:
        return Slist([item for item in self if item is not None])

    def flat_map_option(self, func: Callable[[A], Optional[B]]) -> Slist[B]:
        """Runs the provided function, and filters out the Nones"""
        return self.map(func).flatten_option()

    def upsample_if(self, predicate: Callable[[A], bool], upsample_by: int) -> Slist[A]:
        """Upsamples the list by the given factor if the predicate is true"""
        assert upsample_by > 0
        new_list = Slist[A]()
        for item in self:
            if predicate(item):
                for _ in range(upsample_by):
                    new_list.append(item)
            else:
                new_list.append(item)
        return new_list

    def flatten_list(self: Sequence[Sequence[B]]) -> Slist[B]:
        flat_list: Slist[B] = Slist()
        for sublist in self:
            for item in sublist:
                flat_list.append(item)
        return flat_list

    def enumerated(self) -> Slist[Tuple[int, A]]:
        return Slist(enumerate(self))

    def shuffle(self, seed: Optional[str] = None) -> Slist[A]:
        new = self.copy()
        random.Random(seed).shuffle(new)
        return Slist(new)  # shuffle makes it back into a list instead of Slist

    def choice(
        self,
        seed: Optional[str] = None,
        weights: Optional[List[int]] = None,
    ) -> A:
        if weights:
            return random.Random(seed).choices(self, weights=weights, k=1)[0]
        else:
            return random.Random(seed).choice(self)

    def sample(self, n: int, seed: Optional[str] = None) -> Slist[A]:
        return Slist(random.Random(seed).sample(self, n))

    def for_each(self, func: Callable[[A], None]) -> Slist[A]:
        """Runs an effect on each element, and returns the original list
        e.g. Slist([1,2,3]).foreach(print)"""
        for item in self:
            func(item)
        return self

    def group_by(self, key: Callable[[A], CanHash]) -> Slist[Group[CanHash, Slist[A]]]:
        """
        Groups the list by the given key
        Returns a NamedTuple with attributes
            key: The key that was used to group
            values: The values that were grouped


        Examples
        ----
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
                lambda group: group.map_values(len)
            )
            assert result == Slist(
                [
                    ("cat", 2),
                    ("dog", 1),
                ]
            )
        """
        d: typing.OrderedDict[CanHash, Slist[A]] = OrderedDict()
        for elem in self:
            k = key(elem)
            if k in d:
                d[k].append(elem)
            else:
                d[k] = Slist([elem])
        return Slist(Group(key=key, values=value) for key, value in d.items())

    def map_on_group_values(self: Slist[Group[A, Sequence[B]]], func: Callable[[Sequence[B]], C]) -> Slist[Group[A, C]]:
        # Apply a function on the group's vsalues
        return self.map(lambda group: group.map_values(func))


    def map_on_group_values_list(
        self: Slist[Group[A, Sequence[B]]], func: Callable[[B], C]
    ) -> Slist[Group[A, Sequence[C]]]:
        # If your group's values are a list, and you want to apply a function on each element of the list
        # e.g. Slist([Group(1, [1, 2, 3]), Group(2, [4, 5, 6])]).map_on_group_values_list(lambda x: x + 1)
        # Slist([Group(1, [2, 3, 4]), Group(2, [5, 6, 7])]) 
        return self.map(lambda group: group.map_values(lambda values: Slist(values).map(func)))

    def to_dict(self: Sequence[Tuple[CanHash, B]]) -> typing.Dict[CanHash, B]:
        """
        Transforms a Slist of key value pairs to a dictionary
        >>> Slist([(1, Slist([1, 1])), (2, Slist([2, 2])])).to_dict()
        # Equivalent to
        >>> Slist([1, 1, 2, 2]).group_by(lambda x: x).to_dict()
        {1: Slist([1, 1]), 2: Slist([2, 2])}
        """
        return dict(self)

    def to_set(self) -> typing.Set[A]:
        return set(self)

    @staticmethod
    def from_dict(a_dict: typing.Dict[CanHash, A]) -> Slist[Tuple[CanHash, A]]:
        return Slist(tup for tup in a_dict.items())

    def for_each_enumerate(self, func: Callable[[int, A], None]) -> Slist[A]:
        """Runs an effect on each element, and returns the original list
        e.g. Slist([1,2,3]).foreach(print)"""
        for idx, item in enumerate(self):
            func(idx, item)
        return self

    def max_option(self: Sequence[CanCompare]) -> Optional[CanCompare]:
        return max(self) if self else None

    def max_by(self, key: Callable[[A], CanCompare]) -> Optional[A]:
        return max(self, key=key) if self.length > 0 else None

    def max_by_ordering(self, ordering: Callable[[A, A], bool]) -> Optional[A]:
        theMax: Optional[A] = self.first_option
        for currentItem in self:
            if theMax is not None:
                if ordering(theMax, currentItem):
                    theMax = currentItem
        return theMax

    def min_option(self: Sequence[CanCompare]) -> Optional[CanCompare]:
        return min(self) if self else None

    def min_by(self, key: Callable[[A], CanCompare]) -> Optional[A]:
        return min(self, key=key) if self.length > 0 else None

    def min_by_ordering(self: Slist[CanCompare]) -> Optional[CanCompare]:
        return min(self) if self else None

    def get(self, index: int, or_else: B) -> Union[A, B]:
        try:
            return self.__getitem__(index)
        except IndexError:
            return or_else

    def get_option(self, index: int) -> Optional[A]:
        try:
            return self.__getitem__(index)
        except IndexError:
            return None

    def pairwise(self) -> Slist[Tuple[A, A]]:
        """From more-itertools
        Returns an iterator of paired items, overlapping, from the original"""
        a, b = tee(self)
        next(b, None)
        return Slist(zip(a, b))

    def print_length(self, printer: Callable[[str], None] = print, prefix: str = "Slist Length: ") -> Slist[A]:
        """Prints the length of the list, and returns the original list
        e.g. Slist([1,2,3]).print_length()
        >>> Slist Length: 3
        """
        string = f"{prefix}{len(self)}"
        printer(string)
        return self

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    @property
    def not_empty(self) -> bool:
        return len(self) > 0

    @property
    def length(self) -> int:
        return len(self)

    @property
    def last_option(self) -> Optional[A]:
        try:
            return self.__getitem__(-1)
        except IndexError:
            return None

    @property
    def first_option(self) -> Optional[A]:
        try:
            return self.__getitem__(0)
        except IndexError:
            return None

    @property
    def mode_option(self) -> Optional[A]:
        try:
            return statistics.mode(self)
        except statistics.StatisticsError:
            return None

    def mode_or_raise(self, exception: Exception = RuntimeError("List is empty")) -> A:
        try:
            return statistics.mode(self)
        except statistics.StatisticsError:
            raise exception

    def first_or_raise(self, exception: Exception = RuntimeError("List is empty")) -> A:
        try:
            return self.__getitem__(0)
        except IndexError:
            raise exception

    def last_or_raise(self, exception: Exception = RuntimeError("List is empty")) -> A:
        try:
            return self.__getitem__(-1)
        except IndexError:
            raise exception

    def find_one(self, predicate: Callable[[A], bool]) -> Optional[A]:
        for item in self:
            if predicate(item):
                return item
        return None

    @overload
    def zip(
        self,
        __second: Sequence[B],
        __third: Sequence[C],
        __fourth: Sequence[D],
        __fifth: Sequence[E],
    ) -> Slist[Tuple[A, B, C, D, E]]:
        ...

    @overload
    def zip(self, __second: Sequence[B], __third: Sequence[C], __fourth: Sequence[D]) -> Slist[Tuple[A, B, C, D]]:
        ...

    @overload
    def zip(self, __second: Sequence[B], __third: Sequence[C]) -> Slist[Tuple[A, B, C]]:
        ...

    @overload
    def zip(self, __second: Sequence[B]) -> Slist[Tuple[A, B]]:
        ...

    def zip(self, *args: Sequence[Any]) -> Slist[Any]:
        """Zips the list with the given sequences"""
        all_args = args
        # raise errors if all args are not the same length
        for arg in all_args:
            if len(arg) != len(self):
                raise TypeError(f"Zipping with two different length sequences. {len(self)} != {len(arg)}")
        return Slist(zip(self, *all_args))

    def slice_with_bool(self, bools: Sequence[bool]) -> Slist[A]:
        """Gets elements of the list with a sequence of booleans that are of the same length"""
        return self.zip(bools).flat_map_option(lambda tup: tup[0] if tup[1] is True else None)

    def find_one_or_raise(
        self,
        predicate: Callable[[A], bool],
        exception: Exception = RuntimeError("Failed to find predicate"),
    ) -> A:
        result = self.find_one(predicate=predicate)
        if result is not None:
            return result
        else:
            raise exception

    def find_one_idx(self, predicate: Callable[[A], bool]) -> Optional[int]:
        for idx, item in enumerate(self):
            if predicate(item):
                return idx
        return None

    def find_last_idx(self, predicate: Callable[[A], bool]) -> Optional[int]:
        indexes = []
        for idx, item in enumerate(self):
            if predicate(item):
                indexes.append(idx)
        return indexes[-1] if indexes else None

    def find_one_idx_or_raise(
        self,
        predicate: Callable[[A], bool],
        exception: Exception = RuntimeError("Failed to find predicate"),
    ) -> int:
        result = self.find_one_idx(predicate=predicate)
        if result is not None:
            return result
        else:
            raise exception

    def find_last_idx_or_raise(
        self,
        predicate: Callable[[A], bool],
        exception: Exception = RuntimeError("Failed to find predicate"),
    ) -> int:
        result = self.find_last_idx(predicate=predicate)
        if result is not None:
            return result
        else:
            raise exception

    def take(self, n: int) -> Slist[A]:
        return Slist(self[:n])

    def take_until_exclusive(self, predicate: Callable[[A], bool]) -> Slist[A]:
        """Takes the first elements until the predicate is true.
        Does not include the element that caused the predicate to return true."""
        new: Slist[A] = Slist()
        for x in self:
            if predicate(x):
                break
            else:
                new.append(x)
        return new

    def take_until_inclusive(self, predicate: Callable[[A], bool]) -> Slist[A]:
        """Takes the first elements until the predicate is true.
        Includes the element that caused the predicate to return true."""
        new: Slist[A] = Slist()
        for x in self:
            if predicate(x):
                new.append(x)
                break
            else:
                new.append(x)
        return new

    def sort_by(self, key: Callable[[A], CanCompare], reverse: bool = False) -> Slist[A]:
        new = self.copy()
        return Slist(sorted(new, key=key, reverse=reverse))

    def percentile_by(self, key: Callable[[A], CanCompare], percentile: float) -> A:
        """Gets the element at the given percentile"""
        if percentile < 0 or percentile > 1:
            raise ValueError(f"Percentile must be between 0 and 1. Got {percentile}")
        if self.length == 0:
            raise ValueError("Cannot get percentile of empty list")
        result = self.sort_by(key).get(int(len(self) * percentile), None)
        assert result is not None
        return result

    def median_by(self, key: Callable[[A], CanCompare]) -> A:
        """Gets the median element"""
        if self.length == 0:
            raise ValueError("Cannot get median of empty list")
        return self.percentile_by(key, 0.5)

    def sorted(self: Slist[CanCompare], reverse: bool = False) -> Slist[CanCompare]:
        return self.sort_by(key=identity, reverse=reverse)

    def reversed(self) -> Slist[A]:
        """Returns a new list with the elements in reversed order"""
        return Slist(reversed(self))

    def sort_by_penalise_duplicates(
        self,
        sort_key: Callable[[A], CanCompare],
        duplicate_key: Callable[[A], CanHash],
        reverse: bool = False,
    ) -> Slist[A]:
        """Sort on a given sort key, but penalises duplicate_key such that they will be at the back of the list
        # >>> Slist([6, 5, 4, 3, 2, 1, 1, 1]).sort_by_penalise_duplicates(sort_key=identity, duplicate_key=identity)
        [1, 2, 3, 4, 5, 6, 1, 1]
        """
        non_dupes = Slist[A]()
        dupes = Slist[A]()

        dupes_tracker: set[CanHash] = set()
        for item in self:
            dupe_key = duplicate_key(item)
            if dupe_key in dupes_tracker:
                dupes.append(item)
            else:
                non_dupes.append(item)
                dupes_tracker.add(dupe_key)

        return non_dupes.sort_by(key=sort_key, reverse=reverse) + dupes.sort_by(key=sort_key, reverse=reverse)

    def shuffle_with_penalise_duplicates(
        self,
        duplicate_key: Callable[[A], CanHash],
        seed: Optional[str] = None,
    ) -> Slist[A]:
        """Shuffle, but penalises duplicate_key such that they will be at the back of the list
        # >>> Slist([6, 5, 4, 3, 2, 2, 1, 1, 1]).shuffle_by_penalise_duplicates(duplicate_key=identity)
        [6, 4, 1, 3, 5, 2, 1, 2, 1]
        """
        non_dupes = Slist[A]()
        dupes = Slist[A]()
        shuffled = self.shuffle(seed)

        dupes_tracker: set[CanHash] = set()
        for item in shuffled:
            dupe_key = duplicate_key(item)
            if dupe_key in dupes_tracker:
                dupes.append(item)
            else:
                non_dupes.append(item)
                dupes_tracker.add(dupe_key)

        return non_dupes.shuffle(seed) + dupes.shuffle(seed)

    def __add__(self, other: Slist[B]) -> Slist[Union[A, B]]:  # type: ignore
        return Slist(super().__add__(other))  # type: ignore

    def add(self, other: Slist[B]) -> Slist[Union[A, B]]:
        return self + other  # type: ignore

    def add_one(self, other: B) -> Slist[Union[A, B]]:
        new: Slist[Union[A, B]] = self.copy()  # type: ignore
        new.append(other)
        return new

    @overload  # type: ignore
    def __getitem__(self, i: int) -> A:
        pass

    @overload
    def __getitem__(self, i: slice) -> Slist[A]:
        pass

    def __getitem__(self, i: Union[int, slice]) -> Union[A, Slist[A]]:  # type: ignore
        if isinstance(i, int):
            return super().__getitem__(i)
        else:
            return Slist(super(Slist, self).__getitem__(i))

    def grouped(self, size: int) -> Slist[Slist[A]]:
        """Groups the list into chunks of size `size`"""
        output = Slist[Slist[A]]()
        for i in range(0, self.length, size):
            output.append(self[i : i + size])
        return output

    def window(self, size: int) -> Slist[Slist[A]]:
        """Returns a list of windows of size `size`
        If the list is too small or empty, returns an empty list
        Example:
        >>> Slist([1, 2, 3, 4, 5]).window(3)
        [[1, 2, 3], [2, 3, 4], [3, 4, 5]]

        >>> Slist([1]).window(2)
        []
        """
        output = Slist[Slist[A]]()
        for i in range(0, self.length - size + 1):
            output.append(self[i : i + size])
        return output

    def distinct(self: Sequence[CanHash]) -> Slist[CanHash]:
        """Deduplicates items. Preserves order."""
        return self.distinct_unsafe()  # type: ignore

    def distinct_unsafe(self: Sequence[CanHash]) -> Slist[CanHash]:
        """Deduplicates items. Preserves order.
        To be deprecated in favour of distinct
        Previously was type-unsafe due to mypy bug
        """
        seen = set()
        output = Slist[CanHash]()
        for item in self:
            if item in seen:
                continue
            else:
                seen.add(item)
                output.append(item)
        return output

    def distinct_by(self, key: Callable[[A], CanHash]) -> Slist[A]:
        """Deduplicates a list by a key. Preserves order."""
        seen = set()
        output = Slist[A]()
        for item in self:
            item_hash = key(item)
            if item_hash in seen:
                continue
            else:
                seen.add(item_hash)
                output.append(item)
        return output

    def distinct_item_or_raise(self, key: Callable[[A], CanHash]) -> A:
        """
        Returns the distinct item in the list.
        If the list is empty, raises an error
        If the items in the list are not distinct, raises an error"""
        if not self:
            raise ValueError("Slist is empty")
        distinct = self.distinct_by(key)
        if len(distinct) != 1:
            raise ValueError(f"Slist is not distinct {self}")
        return distinct[0]

    def par_map(self, func: Callable[[A], B], executor: concurrent.futures.Executor) -> Slist[B]:
        """Applies the function to each element using the specified executor. Awaits for all results.
        If executor is ProcessPoolExecutor, make sure the function passed is pickable, e.g. no lambda functions"""
        futures: List[concurrent.futures._base.Future[B]] = [executor.submit(func, item) for item in self]
        results = []
        for fut in futures:
            results.append(fut.result())
        return Slist(results)

    async def par_map_async(self, func: Callable[[A], typing.Awaitable[B]]) -> Slist[B]:
        """Applies the async function to each element. Awaits for all results."""
        return Slist(await asyncio.gather(*[func(item) for item in self]))

    def filter_text_search(self, key: Callable[[A], str], search: List[str]) -> Slist[A]:
        """Filters a list of text with text terms"""

        def matches_search(text: str) -> bool:
            if search:
                search_regex = re.compile("|".join(search), re.IGNORECASE)
                return bool(re.search(search_regex, text))
            else:
                return True  # No filter if search undefined

        return self.filter(predicate=lambda item: matches_search(key(item)))

    def mk_string(self: Sequence[str], sep: str) -> str:
        return sep.join(self)

    @overload
    def sum(self: Sequence[int]) -> int:
        ...

    @overload
    def sum(self: Sequence[float]) -> float:
        ...

    def sum(
        self: Sequence[Union[int, float, bool]],
    ) -> Union[int, float]:
        """Returns 0 when the list is empty"""
        return sum(self)

    def average(
        self: Sequence[Union[int, float, bool]],
    ) -> Optional[float]:
        """Returns None when the list is empty"""
        this = typing.cast(Slist[Union[int, float, bool]], self)
        return this.sum() / this.length if this.length > 0 else None

    def average_or_raise(
        self: Sequence[Union[int, float, bool]],
    ) -> float:
        """Returns None when the list is empty"""
        this = typing.cast(Slist[Union[int, float, bool]], self)
        if this.length == 0:
            raise ValueError("Cannot get average of empty list")
        return this.sum() / this.length

    def standard_deviation(self: Slist[Union[int, float]]) -> Optional[float]:
        """Returns None when the list is empty"""
        return statistics.stdev(self) if self.length > 0 else None

    def standardize(self: Slist[Union[int, float]]) -> Slist[float]:
        """standardize to mean 0, sd 1
        Returns empty list when the list is empty"""
        mean = self.average()
        sd = self.standard_deviation()
        return Slist((x - mean) / sd for x in self) if mean is not None and sd is not None else Slist()

    def fold_left(self, acc: B, func: Callable[[B, A], B]) -> B:
        return reduce(func, self, acc)

    def fold_right(self, acc: B, func: Callable[[A, B], B]) -> B:
        return reduce(lambda a, b: func(b, a), reversed(self), acc)

    def sum_option(self: Sequence[CanAdd]) -> Optional[CanAdd]:
        """Folds left with addition. Returns None if the list is empty"""
        return reduce(lambda a, b: a + b, self) if len(self) > 0 else None

    def sum_or_raise(self: Sequence[CanAdd]) -> CanAdd:
        """Folds left with addition. Raises if the list is empty"""
        assert len(self) > 0, "Cannot fold empty list"
        return reduce(lambda a, b: a + b, self)

    def split_by(self, predicate: Callable[[A], bool]) -> Tuple[Slist[A], Slist[A]]:
        """Splits the list into two lists based on the predicate"""
        left = Slist[A]()
        right = Slist[A]()
        for item in self:
            if predicate(item):
                left.append(item)
            else:
                right.append(item)
        return left, right

    def split_on(self, predicate: Callable[[A], bool]) -> Slist[Slist[A]]:
        """Splits the list into sections based on the predicate,
        items matching the predicate are not included in the output"""
        output = Slist[Slist[A]]()
        current = Slist[A]()
        for item in self:
            if predicate(item):
                output.append(current)
                current = Slist[A]()
            else:
                current.append(item)
        output.append(current)
        return output

    def split_proportion(self, left_proportion: float) -> Tuple[Slist[A], Slist[A]]:
        """Splits the list into two lists based on the left_proportion. 0 < left_proportion < 1"""
        assert 0 < left_proportion < 1, "left_proportion needs to be between 0 and 1"
        left = Slist[A]()
        right = Slist[A]()
        for idx, item in enumerate(self):
            if idx < len(self) * left_proportion:
                left.append(item)
            else:
                right.append(item)
        return left, right

    def split_into_n(self, n: int) -> Slist[Slist[A]]:
        """Splits the list into n lists of roughly equal size"""
        assert n > 0, "n needs to be greater than 0"
        output = Slist[Slist[A]]()
        for i in range(n):
            output.append(Slist[A]())
        for idx, item in enumerate(self):
            output[idx % n].append(item)
        return output

    def copy(self) -> Slist[A]:
        return Slist(super().copy())

    def repeat_until_size(self, size: int) -> Optional[Slist[A]]:
        """
        Repeats the list until it reaches the specified size
        >>> Slist(1, 2, 3).repeat_until_size(5)
        Slist(1, 2, 3, 1, 2)

        Returns None if the list is empty
        >>> Slist().repeat_until_size(5)
        None

        Returns a truncated list if the list is longer than the specified size
        >>> Slist(1, 2, 3).repeat_until_size(2)
        Slist(1, 2)
        """
        assert size > 0, "size needs to be greater than 0"
        if self.is_empty:
            return None
        else:
            new = Slist[A]()
            while True:
                for item in self:
                    if len(new) >= size:
                        return new
                    else:
                        new.append(item)

    def repeat_until_size_or_raise(self, size: int) -> Slist[A]:
        """
        Repeats the list until it reaches the specified size
        >>> Slist(1, 2, 3).repeat_until_size_or_raise(5)
        Slist(1, 2, 3, 1, 2)

        Returns a truncated list if the list is longer than the specified size
        >>> Slist(1, 2, 3).repeat_until_size_or_raise(2)
        Slist(1, 2)

        Throws an exception if the list is empty
        """
        assert size > 0, "size needs to be greater than 0"
        assert not self.is_empty, "input needs to be non empty"
        new = Slist[A]()
        while True:
            for item in self:
                if len(new) >= size:
                    return new
                else:
                    new.append(item)

    def __mul__(self, other: typing.SupportsIndex) -> Slist[A]:
        return Slist(super().__mul__(other))

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler):  # type: ignore
        # only called by pydantic v2
        from pydantic_core import core_schema  # type: ignore

        return core_schema.no_info_after_validator_function(cls, handler(list))
