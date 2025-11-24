from __future__ import annotations

import asyncio
import concurrent.futures
from dataclasses import dataclass
import itertools
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
F = TypeVar("F")

CanCompare = TypeVar("CanCompare", bound="Comparable")
CanHash = TypeVar("CanHash", bound=Hashable)


def identity(x: A) -> A:
    return x


A_co = TypeVar("A_co", covariant=True)
B_co = TypeVar("B_co", covariant=True)


class Group(NamedTuple, Generic[A_co, B_co]):
    """This is a NamedTuple so that you can easily access the key and values"""

    key: A_co
    values: B_co

    def map_key(self, func: Callable[[A_co], C]) -> Group[C, B_co]:
        return Group(func(self.key), self.values)

    def map_values(self, func: Callable[[B_co], C]) -> Group[A_co, C]:
        return Group(self.key, func(self.values))


class Addable(Protocol):
    def __add__(self: A, other: A, /) -> A: ...


CanAdd = TypeVar("CanAdd", bound=Addable)


@dataclass(frozen=True)
class AverageStats:
    average: float
    standard_deviation: float
    upper_confidence_interval_95: float
    lower_confidence_interval_95: float
    average_plus_minus_95: float
    count: int

    def __str__(self) -> str:
        return f"Average: {self.average}, SD: {self.standard_deviation}, 95% CI: ({self.lower_confidence_interval_95}, {self.upper_confidence_interval_95})"


class Comparable(Protocol):
    def __lt__(self: CanCompare, other: CanCompare, /) -> bool: ...

    def __gt__(self: CanCompare, other: CanCompare, /) -> bool: ...

    def __le__(self: CanCompare, other: CanCompare, /) -> bool: ...

    def __ge__(self: CanCompare, other: CanCompare, /) -> bool: ...


class Slist(List[A]):
    @staticmethod
    def one(element: A) -> Slist[A]:
        """Create a new Slist with a single element.

        Parameters
        ----------
        element : A
            The element to create the list with

        Returns
        -------
        Slist[A]
            A new Slist containing only the given element

        Examples
        --------
        >>> Slist.one(5)
        Slist([5])
        """
        return Slist([element])

    @staticmethod
    def one_option(element: Optional[A]) -> Slist[A]:
        """Create a Slist with one element if it exists, otherwise empty list.

        Equal to ``Slist.one(element).flatten_option()``

        Parameters
        ----------
        element : Optional[A]
            The element to create the list with, if it exists

        Returns
        -------
        Slist[A]
            A new Slist containing the element if it exists, otherwise empty

        Examples
        --------
        >>> Slist.one_option(5)
        Slist([5])
        >>> Slist.one_option(None)
        Slist([])
        """
        return Slist([element]) if element is not None else Slist()

    def any(self, predicate: Callable[[A], bool]) -> bool:
        """Check if any element satisfies the predicate.

        Parameters
        ----------
        predicate : Callable[[A], bool]
            Function that takes an element and returns True/False

        Returns
        -------
        bool
            True if any element satisfies the predicate, False otherwise

        Examples
        --------
        >>> Slist([1, 2, 3, 4]).any(lambda x: x > 3)
        True
        >>> Slist([1, 2, 3]).any(lambda x: x > 3)
        False
        """
        for x in self:
            if predicate(x):
                return True
        return False

    def all(self, predicate: Callable[[A], bool]) -> bool:
        """Check if all elements satisfy the predicate.

        Parameters
        ----------
        predicate : Callable[[A], bool]
            Function that takes an element and returns True/False

        Returns
        -------
        bool
            True if all elements satisfy the predicate, False otherwise

        Examples
        --------
        >>> Slist([2, 4, 6]).all(lambda x: x % 2 == 0)
        True
        >>> Slist([2, 3, 4]).all(lambda x: x % 2 == 0)
        False
        """
        for x in self:
            if not predicate(x):
                return False
        return True

    def filter(self, predicate: Callable[[A], bool]) -> Slist[A]:
        """Create a new Slist with only elements that satisfy the predicate.

        Parameters
        ----------
        predicate : Callable[[A], bool]
            Function that takes an element and returns True/False

        Returns
        -------
        Slist[A]
            A new Slist containing only elements that satisfy the predicate

        Examples
        --------
        >>> Slist([1, 2, 3, 4]).filter(lambda x: x % 2 == 0)
        Slist([2, 4])
        """
        return Slist(filter(predicate, self))

    def map(self, func: Callable[[A], B]) -> Slist[B]:
        """Transform each element using the given function.

        Parameters
        ----------
        func : Callable[[A], B]
            Function to apply to each element

        Returns
        -------
        Slist[B]
            A new Slist with transformed elements

        Examples
        --------
        >>> Slist([1, 2, 3]).map(lambda x: x * 2)
        Slist([2, 4, 6])
        """
        return Slist(func(item) for item in self)

    @overload
    def product(self: Sequence[A], other: Sequence[B], /) -> Slist[Tuple[A, B]]: ...

    @overload
    def product(self: Sequence[A], other: Sequence[B], other1: Sequence[C], /) -> Slist[Tuple[A, B, C]]: ...

    @overload
    def product(
        self: Sequence[A], other: Sequence[B], other1: Sequence[C], other2: Sequence[D], /
    ) -> Slist[Tuple[A, B, C, D]]: ...

    @overload
    def product(
        self: Sequence[A], other: Sequence[B], other1: Sequence[C], other2: Sequence[D], other3: Sequence[E], /
    ) -> Slist[Tuple[A, B, C, D, E]]: ...

    @overload
    def product(
        self: Sequence[A],
        other: Sequence[B],
        other1: Sequence[C],
        other2: Sequence[D],
        other3: Sequence[E],
        other4: Sequence[F],
        /,
    ) -> Slist[Tuple[A, B, C, D, E, F]]: ...

    def product(self: Sequence[A], *others: Sequence[Any]) -> Slist[Tuple[Any, ...]]:
        """Compute the cartesian product with other sequences.

        Parameters
        ----------
        *others : Sequence[Any]
            The sequences to compute the product with

        Returns
        -------
        Slist[Tuple[Any, ...]]
            A new Slist containing tuples of all combinations

        Examples
        --------
        >>> Slist([1, 2]).product(['a', 'b'])
        Slist([(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')])
        """
        return Slist(itertools.product(self, *others))

    def map_2(self: Sequence[Tuple[B, C]], func: Callable[[B, C], D]) -> Slist[D]:
        """Map a function over a sequence of 2-tuples.

        Parameters
        ----------
        func : Callable[[B, C], D]
            Function that takes two arguments and returns a value

        Returns
        -------
        Slist[D]
            A new Slist with the results of applying func to each tuple

        Examples
        --------
        >>> pairs = Slist([(1, 2), (3, 4)])
        >>> pairs.map_2(lambda x, y: x + y)
        Slist([3, 7])
        """
        return Slist(func(b, c) for b, c in self)

    def map_enumerate(self, func: Callable[[int, A], B]) -> Slist[B]:
        """Map a function over the list with indices.

        Parameters
        ----------
        func : Callable[[int, A], B]
            Function that takes an index and value and returns a new value

        Returns
        -------
        Slist[B]
            A new Slist with the results of applying func to each (index, value) pair

        Examples
        --------
        >>> Slist(['a', 'b', 'c']).map_enumerate(lambda i, x: f"{i}:{x}")
        Slist(['0:a', '1:b', '2:c'])
        """
        return Slist(func(idx, item) for idx, item in enumerate(self))

    def flatten_option(self: Sequence[Optional[B]]) -> Slist[B]:
        """Remove None values from a sequence of optional values.

        Parameters
        ----------
        self : Sequence[Optional[B]]
            A sequence containing optional values

        Returns
        -------
        Slist[B]
            A new Slist with all non-None values

        Examples
        --------
        >>> Slist([1, None, 3, None, 5]).flatten_option()
        Slist([1, 3, 5])
        """
        return Slist([item for item in self if item is not None])

    def flat_map_option(self, func: Callable[[A], Optional[B]]) -> Slist[B]:
        """Apply a function that returns optional values and filter out Nones.

        Parameters
        ----------
        func : Callable[[A], Optional[B]]
            Function that takes a value and returns an optional value

        Returns
        -------
        Slist[B]
            A new Slist with all non-None results of applying func

        Examples
        --------
        >>> def safe_sqrt(x: float) -> Optional[float]:
        ...     return x ** 0.5 if x >= 0 else None
        >>> Slist([4, -1, 9, -4, 16]).flat_map_option(safe_sqrt)
        Slist([2.0, 3.0, 4.0])
        """
        return self.map(func).flatten_option()

    def upsample_if(self, predicate: Callable[[A], bool], upsample_by: int) -> Slist[A]:
        """Repeat elements that satisfy a predicate.

        Parameters
        ----------
        predicate : Callable[[A], bool]
            Function that determines which elements to upsample
        upsample_by : int
            Number of times to repeat each matching element

        Returns
        -------
        Slist[A]
            A new Slist with matching elements repeated

        Examples
        --------
        >>> numbers = Slist([1, 2, 3, 4])
        >>> numbers.upsample_if(lambda x: x % 2 == 0, upsample_by=2)
        Slist([1, 2, 2, 3, 4, 4])
        """
        assert upsample_by > 0, "upsample_by must be positive"
        new_list = Slist[A]()
        for item in self:
            if predicate(item):
                for _ in range(upsample_by):
                    new_list.append(item)
            else:
                new_list.append(item)
        return new_list

    def flatten_list(self: Sequence[Sequence[B]]) -> Slist[B]:
        """Flatten a sequence of sequences into a single list.

        Parameters
        ----------
        self : Sequence[Sequence[B]]
            A sequence of sequences to flatten

        Returns
        -------
        Slist[B]
            A new Slist with all elements from all sequences

        Examples
        --------
        >>> nested = Slist([[1, 2], [3, 4], [5, 6]])
        >>> nested.flatten_list()
        Slist([1, 2, 3, 4, 5, 6])
        """
        flat_list: Slist[B] = Slist()
        for sublist in self:
            for item in sublist:
                flat_list.append(item)
        return flat_list

    def enumerated(self) -> Slist[Tuple[int, A]]:
        """Create a list of tuples containing indices and values.

        Returns
        -------
        Slist[Tuple[int, A]]
            A new Slist of (index, value) tuples

        Examples
        --------
        >>> Slist(['a', 'b', 'c']).enumerated()
        Slist([(0, 'a'), (1, 'b'), (2, 'c')])
        """
        return Slist(enumerate(self))

    def shuffle(self, seed: Optional[str] = None) -> Slist[A]:
        """Create a new randomly shuffled list.

        Parameters
        ----------
        seed : Optional[str], optional
            Random seed for reproducibility, by default None

        Returns
        -------
        Slist[A]
            A new Slist with elements in random order

        Examples
        --------
        >>> Slist([1, 2, 3, 4]).shuffle(seed="42")  # Reproducible shuffle
        Slist([2, 4, 1, 3])
        """
        new = self.copy()
        random.Random(seed).shuffle(new)
        return Slist(new)

    def choice(
        self,
        seed: Optional[str] = None,
        weights: Optional[List[int]] = None,
    ) -> A:
        """Randomly select an element from the list.

        Parameters
        ----------
        seed : Optional[str], optional
            Random seed for reproducibility, by default None
        weights : Optional[List[int]], optional
            List of weights for weighted random selection, by default None

        Returns
        -------
        A
            A randomly selected element

        Examples
        --------
        >>> Slist([1, 2, 3, 4]).choice(seed="42")  # Reproducible choice
        2
        >>> Slist([1, 2, 3]).choice(weights=[1, 2, 1])  # Weighted choice
        2  # More likely to select 2 due to higher weight
        """
        if weights:
            return random.Random(seed).choices(self, weights=weights, k=1)[0]
        else:
            return random.Random(seed).choice(self)

    def sample(self, n: int, seed: Optional[str] = None) -> Slist[A]:
        """Randomly sample n elements from the list without replacement.

        Parameters
        ----------
        n : int
            Number of elements to sample
        seed : Optional[str], optional
            Random seed for reproducibility, by default None

        Returns
        -------
        Slist[A]
            A new Slist with n randomly selected elements

        Examples
        --------
        >>> Slist([1, 2, 3, 4, 5]).sample(3, seed="42")
        Slist([2, 4, 1])
        """
        return Slist(random.Random(seed).sample(self, n))

    def for_each(self, func: Callable[[A], None]) -> Slist[A]:
        """Apply a side-effect function to each element and return the original list.

        Parameters
        ----------
        func : Callable[[A], None]
            Function to apply to each element for its side effects

        Returns
        -------
        Slist[A]
            The original list, unchanged

        Examples
        --------
        >>> nums = Slist([1, 2, 3])
        >>> nums.for_each(print)  # Prints each number
        1
        2
        3
        >>> nums  # Original list is unchanged
        Slist([1, 2, 3])
        """
        for item in self:
            func(item)
        return self

    def group_by(self, key: Callable[[A], CanHash]) -> Slist[Group[CanHash, Slist[A]]]:
        """Group elements by a key function.

        Parameters
        ----------
        key : Callable[[A], CanHash]
            Function to compute the group key for each element

        Returns
        -------
        Slist[Group[CanHash, Slist[A]]]
            A new Slist of Groups, where each Group contains:
                - key: The grouping key
                - values: Slist of elements in that group

        Examples
        --------
        >>> numbers = Slist([1, 2, 3, 4])
        >>> groups = numbers.group_by(lambda x: x % 2)  # Group by even/odd
        >>> groups.map(lambda g: (g.key, list(g.values)))
        Slist([(1, [1, 3]), (0, [2, 4])])
        """
        d: typing.OrderedDict[CanHash, Slist[A]] = OrderedDict()
        for elem in self:
            k = key(elem)
            if k in d:
                d[k].append(elem)
            else:
                d[k] = Slist([elem])
        return Slist(Group(key=key, values=value) for key, value in d.items())

    @overload
    def ungroup(self: Slist[Group[Any, Slist[C]]]) -> Slist[C]: ...

    @overload
    def ungroup(self: Slist[Group[Any, Sequence[C]]]) -> Slist[C]: ...

    def ungroup(self: Slist[Group[Any, Slist[C]]] | Slist[Group[Any, Sequence[C]]]) -> Slist[C]:
        """Convert a list of groups back into a flat list of values.

        Parameters
        ----------
        self : Slist[Group[Any, Slist[C]]] | Slist[Group[Any, Sequence[C]]]
            A list of groups to ungroup

        Returns
        -------
        Slist[C]
            A flat list containing all values from all groups

        Examples
        --------
        >>> groups = Slist([Group(0, [1, 2]), Group(1, [3, 4])])
        >>> groups.ungroup()
        Slist([1, 2, 3, 4])
        """
        casted: Slist[Group[Any, Slist[C]]] = self  # type: ignore
        return casted.map_2(lambda _, values: values).flatten_list()

    def map_on_group_values(self: Slist[Group[B, Slist[C]]], func: Callable[[Slist[C]], D]) -> Slist[Group[B, D]]:
        """Apply a function to the values of each group.

        Parameters
        ----------
        func : Callable[[Slist[C]], D]
            Function to apply to each group's values

        Returns
        -------
        Slist[Group[B, D]]
            A new list of groups with transformed values

        Examples
        --------
        >>> groups = Slist([1, 2, 3, 4]).group_by(lambda x: x % 2)
        >>> groups.map_on_group_values(lambda values: sum(values))
        Slist([Group(key=1, values=4), Group(key=0, values=6)])
        """
        return self.map(lambda group: group.map_values(func))

    def map_on_group_values_list(
        self: Slist[Group[A, Sequence[B]]], func: Callable[[B], C]
    ) -> Slist[Group[A, Sequence[C]]]:
        """Apply a function to each element in each group's values.

        Parameters
        ----------
        func : Callable[[B], C]
            Function to apply to each element in each group's values

        Returns
        -------
        Slist[Group[A, Sequence[C]]]
            A new list of groups with transformed elements

        Examples
        --------
        >>> groups = Slist([Group(1, [1, 2]), Group(2, [3, 4])])
        >>> groups.map_on_group_values_list(lambda x: x * 2)
        Slist([Group(1, [2, 4]), Group(2, [6, 8])])
        """
        return self.map(lambda group: group.map_values(lambda values: Slist(values).map(func)))

    def value_counts(self, key: Callable[[A], CanHash], sort: bool = True) -> Slist[Group[CanHash, int]]:
        """Count occurrences of each unique value or key-derived value.

        Parameters
        ----------
        key : Callable[[A], CanHash]
            Function to extract the value to count by
        sort : bool, default=True
            If True, sorts the results by count in descending order

        Returns
        -------
        Slist[Group[CanHash, int]]
            A list of groups with keys and their counts

        Examples
        --------
        >>> Slist(['apple', 'banana', 'cherry']).value_counts(key=lambda x: x)
        Slist([Group(key='apple', values=1), Group(key='banana', values=1), Group(key='cherry', values=1)])
        """
        result = self.group_by(key).map_on_group_values(len)
        if sort:
            return result.sort_by(key=lambda group: group.values, reverse=True)
        return result

    def value_percentage(self, key: Callable[[A], CanHash], sort: bool = True) -> Slist[Group[CanHash, float]]:
        """Count occurrences of each unique value or key-derived value.

        Parameters
        ----------
        key : Callable[[A], CanHash]
            Function to extract the value to count by
        sort : bool, default=True
            If True, sorts the results by percentage in descending order

        Returns
        -------
        Slist[Group[CanHash, float]]
            A list of groups with keys and their percentage of total

        Examples
        --------
        >>> Slist(['a', 'a', 'b']).value_percentage(key=lambda x: x)
        Slist([Group(key='a', values=0.6666666666666666), Group(key='b', values=0.3333333333333333)])
        """
        total = len(self)
        if total == 0:
            return Slist()

        counts = self.value_counts(key, sort=False)
        result = counts.map(lambda group: Group(key=group.key, values=group.values / total))  # type: ignore

        if sort:
            return result.sort_by(key=lambda group: group.values, reverse=True)
        return result

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
        """
        Convert the Slist to a set.
        """
        return set(self)

    @staticmethod
    def from_dict(a_dict: typing.Dict[CanHash, A]) -> Slist[Tuple[CanHash, A]]:
        """Convert a dictionary to a Slist of tuple values.

        Parameters
        ----------
        a_dict : Dict[CanHash, A]
            Dictionary to convert

        Returns
        -------
        Slist[Tuple[CanHash, A]]
            List of key-value tuples from the dictionary

        Examples
        --------
        >>> Slist.from_dict({1: 'a', 2: 'b'})
        Slist([(1, 'a'), (2, 'b')])
        """
        return Slist(tup for tup in a_dict.items())

    def for_each_enumerate(self, func: Callable[[int, A], None]) -> Slist[A]:
        """Apply a side-effect function to each element with its index.

        Parameters
        ----------
        func : Callable[[int, A], None]
            Function taking an index and value, applied for side effects

        Returns
        -------
        Slist[A]
            The original list, unchanged

        Examples
        --------
        >>> nums = Slist(['a', 'b', 'c'])
        >>> nums.for_each_enumerate(lambda i, x: print(f"{i}: {x}"))
        0: a
        1: b
        2: c
        >>> nums  # Original list is unchanged
        Slist(['a', 'b', 'c'])
        """
        for idx, item in enumerate(self):
            func(idx, item)
        return self

    def max_option(self: Sequence[CanCompare]) -> Optional[CanCompare]:
        """Get the maximum element if it exists.

        Returns
        -------
        Optional[CanCompare]
            Maximum element, or None if list is empty

        Examples
        --------
        >>> Slist([1, 3, 2]).max_option()
        3
        >>> Slist([]).max_option()
        None
        """
        return max(self) if self else None

    def max_by(self, key: Callable[[A], CanCompare]) -> Optional[A]:
        """Get the element with maximum value by key function.

        Parameters
        ----------
        key : Callable[[A], CanCompare]
            Function to compute comparison value for each element

        Returns
        -------
        Optional[A]
            Element with maximum key value, or None if list is empty

        Examples
        --------
        >>> Slist(['a', 'bbb', 'cc']).max_by(len)
        'bbb'
        >>> Slist([]).max_by(len)
        None
        """
        return max(self, key=key) if self.length > 0 else None

    def max_by_ordering(self, ordering: Callable[[A, A], bool]) -> Optional[A]:
        """Get maximum element using custom ordering function.

        Parameters
        ----------
        ordering : Callable[[A, A], bool]
            Function that returns True if first argument should be considered larger

        Returns
        -------
        Optional[A]
            Maximum element by ordering, or None if list is empty

        Examples
        --------
        >>> # Custom ordering: consider numbers closer to 10 as "larger"
        >>> nums = Slist([1, 5, 8, 15])
        >>> nums.max_by_ordering(lambda x, y: abs(x-10) < abs(y-10))
        8
        """
        theMax: Optional[A] = self.first_option
        for currentItem in self:
            if theMax is not None:
                if ordering(theMax, currentItem):
                    theMax = currentItem
        return theMax

    def min_option(self: Sequence[CanCompare]) -> Optional[CanCompare]:
        """Get the minimum element if it exists.

        Returns
        -------
        Optional[CanCompare]
            Minimum element, or None if list is empty

        Examples
        --------
        >>> Slist([3, 1, 2]).min_option()
        1
        >>> Slist([]).min_option()
        None
        """
        return min(self) if self else None

    def min_by(self, key: Callable[[A], CanCompare]) -> Optional[A]:
        """Get the element with minimum value by key function.

        Parameters
        ----------
        key : Callable[[A], CanCompare]
            Function to compute comparison value for each element

        Returns
        -------
        Optional[A]
            Element with minimum key value, or None if list is empty

        Examples
        --------
        >>> Slist(['aaa', 'b', 'cc']).min_by(len)
        'b'
        >>> Slist([]).min_by(len)
        None
        """
        return min(self, key=key) if self.length > 0 else None

    def min_by_ordering(self: Slist[CanCompare]) -> Optional[CanCompare]:
        """Get minimum element using default ordering.

        Returns
        -------
        Optional[CanCompare]
            Minimum element, or None if list is empty

        Examples
        --------
        >>> Slist([3, 1, 2]).min_by_ordering()
        1
        >>> Slist([]).min_by_ordering()
        None
        """
        return min(self) if self else None

    def get(self, index: int, or_else: B) -> Union[A, B]:
        """Get element at index with fallback value.

        Parameters
        ----------
        index : int
            Index to get element from
        or_else : B
            Value to return if index is out of bounds

        Returns
        -------
        Union[A, B]
            Element at index if it exists, otherwise or_else value

        Examples
        --------
        >>> Slist([1, 2, 3]).get(1, -1)
        2
        >>> Slist([1, 2, 3]).get(5, -1)
        -1
        """
        try:
            return self.__getitem__(index)
        except IndexError:
            return or_else

    def get_option(self, index: int) -> Optional[A]:
        """Get element at index if it exists.

        Parameters
        ----------
        index : int
            Index to get element from

        Returns
        -------
        Optional[A]
            Element at index if it exists, otherwise None

        Examples
        --------
        >>> Slist([1, 2, 3]).get_option(1)
        2
        >>> Slist([1, 2, 3]).get_option(5)
        None
        """
        try:
            return self.__getitem__(index)
        except IndexError:
            return None

    def pairwise(self) -> Slist[Tuple[A, A]]:
        """Return overlapping pairs of consecutive elements.

        Returns
        -------
        Slist[Tuple[A, A]]
            List of tuples containing consecutive overlapping pairs

        Examples
        --------
        >>> Slist([1, 2, 3, 4]).pairwise()
        Slist([(1, 2), (2, 3), (3, 4)])
        >>> Slist([1]).pairwise()
        Slist([])
        >>> Slist([]).pairwise()
        Slist([])

        Notes
        -----
        Inspired by more-itertools pairwise function. Creates an iterator of
        overlapping pairs from the input sequence.
        """
        a, b = tee(self)
        next(b, None)
        return Slist(zip(a, b))

    def print_length(self, printer: Callable[[str], None] = print, prefix: str = "Slist Length: ") -> Slist[A]:
        """Print the length of the list and return the original list.

        Parameters
        ----------
        printer : Callable[[str], None], optional
            Function to print the output, by default print
        prefix : str, optional
            Prefix string before the length, by default "Slist Length: "

        Returns
        -------
        Slist[A]
            The original list unchanged

        Examples
        --------
        >>> Slist([1,2,3]).print_length()
        Slist Length: 3
        Slist([1, 2, 3])
        """
        string = f"{prefix}{len(self)}"
        printer(string)
        return self

    @property
    def is_empty(self) -> bool:
        """Check if the list is empty.

        Returns
        -------
        bool
            True if the list has no elements

        Examples
        --------
        >>> Slist([]).is_empty
        True
        >>> Slist([1]).is_empty
        False
        """
        return len(self) == 0

    @property
    def not_empty(self) -> bool:
        """Check if the list has any elements.

        Returns
        -------
        bool
            True if the list has at least one element

        Examples
        --------
        >>> Slist([1]).not_empty
        True
        >>> Slist([]).not_empty
        False
        """
        return len(self) > 0

    @property
    def length(self) -> int:
        """Get the number of elements in the list.

        Returns
        -------
        int
            Number of elements

        Examples
        --------
        >>> Slist([1, 2, 3]).length
        3
        """
        return len(self)

    @property
    def last_option(self) -> Optional[A]:
        """Get the last element if it exists.

        Returns
        -------
        Optional[A]
            Last element, or None if list is empty

        Examples
        --------
        >>> Slist([1, 2, 3]).last_option
        3
        >>> Slist([]).last_option
        None
        """
        try:
            return self.__getitem__(-1)
        except IndexError:
            return None

    @property
    def first_option(self) -> Optional[A]:
        """Get the first element if it exists.

        Returns
        -------
        Optional[A]
            First element, or None if list is empty

        Examples
        --------
        >>> Slist([1, 2, 3]).first_option
        1
        >>> Slist([]).first_option
        None
        """
        try:
            return self.__getitem__(0)
        except IndexError:
            return None

    @property
    def mode_option(self) -> Optional[A]:
        """Get the most common element if it exists.

        Returns
        -------
        Optional[A]
            Most frequent element, or None if list is empty or has no unique mode

        Examples
        --------
        >>> Slist([1, 2, 2, 3]).mode_option
        2
        >>> Slist([1, 1, 2, 2]).mode_option  # No unique mode
        None
        >>> Slist([]).mode_option
        None
        """
        try:
            return statistics.mode(self)
        except statistics.StatisticsError:
            return None

    def mode_or_raise(self, exception: Exception = RuntimeError("List is empty")) -> A:
        """Get the most common element or raise an exception.

        Parameters
        ----------
        exception : Exception, optional
            Exception to raise if no mode exists, by default RuntimeError("List is empty")

        Returns
        -------
        A
            Most frequent element

        Raises
        ------
        Exception
            If list is empty or has no unique mode

        Examples
        --------
        >>> Slist([1, 2, 2, 3]).mode_or_raise()
        2
        >>> try:
        ...     Slist([]).mode_or_raise()
        ... except RuntimeError as e:
        ...     print(str(e))
        List is empty
        """
        try:
            return statistics.mode(self)
        except statistics.StatisticsError:
            raise exception

    def first_or_raise(self, exception: Exception = RuntimeError("List is empty")) -> A:
        """Get the first element or raise an exception.

        Parameters
        ----------
        exception : Exception, optional
            Exception to raise if list is empty, by default RuntimeError("List is empty")

        Returns
        -------
        A
            First element

        Raises
        ------
        Exception
            If list is empty

        Examples
        --------
        >>> Slist([1, 2, 3]).first_or_raise()
        1
        >>> try:
        ...     Slist([]).first_or_raise()
        ... except RuntimeError as e:
        ...     print(str(e))
        List is empty
        """
        try:
            return self.__getitem__(0)
        except IndexError:
            raise exception

    def last_or_raise(self, exception: Exception = RuntimeError("List is empty")) -> A:
        """Get the last element or raise an exception.

        Parameters
        ----------
        exception : Exception, optional
            Exception to raise if list is empty, by default RuntimeError("List is empty")

        Returns
        -------
        A
            Last element

        Raises
        ------
        Exception
            If list is empty

        Examples
        --------
        >>> Slist([1, 2, 3]).last_or_raise()
        3
        >>> try:
        ...     Slist([]).last_or_raise()
        ... except RuntimeError as e:
        ...     print(str(e))
        List is empty
        """
        try:
            return self.__getitem__(-1)
        except IndexError:
            raise exception

    def find_one(self, predicate: Callable[[A], bool]) -> Optional[A]:
        """Find first element that satisfies a predicate.

        Parameters
        ----------
        predicate : Callable[[A], bool]
            Function that returns True for the desired element

        Returns
        -------
        Optional[A]
            First matching element, or None if no match found

        Examples
        --------
        >>> Slist([1, 2, 3, 4]).find_one(lambda x: x > 2)
        3
        >>> Slist([1, 2, 3]).find_one(lambda x: x > 5)
        None
        """
        for item in self:
            if predicate(item):
                return item
        return None

    def find_one_idx(self, predicate: Callable[[A], bool]) -> Optional[int]:
        """Find index of first element that satisfies a predicate.

        Parameters
        ----------
        predicate : Callable[[A], bool]
            Function that returns True for the desired element

        Returns
        -------
        Optional[int]
            Index of first matching element, or None if no match found

        Examples
        --------
        >>> Slist([1, 2, 3, 4]).find_one_idx(lambda x: x > 2)
        2
        >>> Slist([1, 2, 3]).find_one_idx(lambda x: x > 5)
        None
        """
        for idx, item in enumerate(self):
            if predicate(item):
                return idx
        return None

    def find_last_idx(self, predicate: Callable[[A], bool]) -> Optional[int]:
        """Find index of last element that satisfies a predicate.

        Parameters
        ----------
        predicate : Callable[[A], bool]
            Function that returns True for the desired element

        Returns
        -------
        Optional[int]
            Index of last matching element, or None if no match found

        Examples
        --------
        >>> Slist([1, 2, 3, 2, 1]).find_last_idx(lambda x: x == 2)
        3
        >>> Slist([1, 2, 3]).find_last_idx(lambda x: x > 5)
        None
        """
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
        """Find index of first element that satisfies a predicate or raise exception.

        Parameters
        ----------
        predicate : Callable[[A], bool]
            Function that returns True for the desired element
        exception : Exception, optional
            Exception to raise if no match found, by default RuntimeError("Failed to find predicate")

        Returns
        -------
        int
            Index of first matching element

        Raises
        ------
        Exception
            If no matching element is found

        Examples
        --------
        >>> Slist([1, 2, 3, 4]).find_one_idx_or_raise(lambda x: x > 2)
        2
        >>> try:
        ...     Slist([1, 2, 3]).find_one_idx_or_raise(lambda x: x > 5)
        ... except RuntimeError as e:
        ...     print(str(e))
        Failed to find predicate
        """
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
        """Find index of last element that satisfies a predicate or raise exception.

        Parameters
        ----------
        predicate : Callable[[A], bool]
            Function that returns True for the desired element
        exception : Exception, optional
            Exception to raise if no match found, by default RuntimeError("Failed to find predicate")

        Returns
        -------
        int
            Index of last matching element

        Raises
        ------
        Exception
            If no matching element is found

        Examples
        --------
        >>> Slist([1, 2, 3, 2, 1]).find_last_idx_or_raise(lambda x: x == 2)
        3
        >>> try:
        ...     Slist([1, 2, 3]).find_last_idx_or_raise(lambda x: x > 5)
        ... except RuntimeError as e:
        ...     print(str(e))
        Failed to find predicate
        """
        result = self.find_last_idx(predicate=predicate)
        if result is not None:
            return result
        else:
            raise exception

    def take(self, n: int) -> Slist[A]:
        return Slist(self[:n])

    def take_or_raise(self, n: int) -> Slist[A]:
        # raises if we end up having less elements than n
        if len(self) < n:
            raise ValueError(f"Cannot take {n} elements from a list of length {len(self)}")
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

    def __add__(self, other: Sequence[B]) -> Slist[Union[A, B]]:  # type: ignore
        return Slist(super().__add__(other))  # type: ignore

    def add(self, other: Sequence[B]) -> Slist[Union[A, B]]:
        return self + other

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
        output: Slist[Slist[A]] = Slist()
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
        output: Slist[Slist[A]] = Slist()
        for i in range(0, self.length - size + 1):
            output.append(self[i : i + size])
        return output

    def distinct(self: Sequence[CanHash]) -> Slist[CanHash]:
        """Remove duplicate elements while preserving order.

        Returns
        -------
        Slist[CanHash]
            A new list with duplicates removed, maintaining original order

        Examples
        --------
        >>> Slist([1, 2, 2, 3, 1, 4]).distinct()
        Slist([1, 2, 3, 4])
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
        """Remove duplicates based on a key function while preserving order.

        Parameters
        ----------
        key : Callable[[A], CanHash]
            Function to compute the unique key for each element

        Returns
        -------
        Slist[A]
            A new list with duplicates removed, maintaining original order

        Examples
        --------
        >>> data = Slist([(1, 'a'), (2, 'b'), (1, 'c')])
        >>> data.distinct_by(lambda x: x[0])  # Distinct by first element
        Slist([(1, 'a'), (2, 'b')])
        """
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
        """Get the single unique item by a key function.

        Raises ValueError if the list is empty or contains multiple distinct items.

        Parameters
        ----------
        key : Callable[[A], CanHash]
            Function to compute the unique key for each element

        Returns
        -------
        A
            The single unique item

        Raises
        ------
        ValueError
            If the list is empty or contains multiple distinct items

        Examples
        --------
        >>> Slist([1, 1, 1]).distinct_item_or_raise(lambda x: x)
        1
        >>> try:
        ...     Slist([1, 2, 1]).distinct_item_or_raise(lambda x: x)
        ... except ValueError as e:
        ...     print(str(e))
        Slist is not distinct [1, 2, 1]
        """
        if not self:
            raise ValueError("Slist is empty")
        distinct = self.distinct_by(key)
        if len(distinct) != 1:
            raise ValueError(f"Slist is not distinct {self}")
        return distinct[0]

    def par_map(self, func: Callable[[A], B], executor: concurrent.futures.Executor) -> Slist[B]:
        """Apply a function to each element in parallel using an executor.

        Parameters
        ----------
        func : Callable[[A], B]
            Function to apply to each element. Must be picklable if using ProcessPoolExecutor
        executor : concurrent.futures.Executor
            The executor to use for parallel execution

        Returns
        -------
        Slist[B]
            A new list with the results of applying func to each element

        Examples
        --------
        >>> from concurrent.futures import ThreadPoolExecutor
        >>> with ThreadPoolExecutor() as exe:
        ...     Slist([1, 2, 3]).par_map(lambda x: x * 2, exe)
        Slist([2, 4, 6])

        Notes
        -----
        If using ProcessPoolExecutor, the function must be picklable (e.g., no lambda functions)
        """
        futures: List[concurrent.futures._base.Future[B]] = [executor.submit(func, item) for item in self]
        results = []
        for fut in futures:
            results.append(fut.result())
        return Slist(results)

    async def par_map_async(
        self, func: Callable[[A], typing.Awaitable[B]], max_par: int | None = None, tqdm: bool = False
    ) -> Slist[B]:
        """Asynchronously apply a function to each element with optional parallelism limit.

        Parameters
        ----------
        func : Callable[[A], Awaitable[B]]
            Async function to apply to each element
        max_par : int | None, optional
            Maximum number of parallel operations, by default None
        tqdm : bool, optional
            Whether to show a progress bar, by default False

        Returns
        -------
        Slist[B]
            A new Slist with the transformed elements

        Examples
        --------
        >>> async def slow_double(x):
        ...     await asyncio.sleep(0.1)
        ...     return x * 2
        >>> await Slist([1, 2, 3]).par_map_async(slow_double, max_par=2)
        Slist([2, 4, 6])
        """
        if max_par is None:
            if tqdm:
                import tqdm as tqdm_module

                tqdm_counter = tqdm_module.tqdm(total=len(self))

                async def func_with_tqdm(item: A) -> B:
                    result = await func(item)
                    tqdm_counter.update(1)
                    return result

                return Slist(await asyncio.gather(*[func_with_tqdm(item) for item in self]))
            else:
                # todo: clean up branching
                return Slist(await asyncio.gather(*[func(item) for item in self]))

        else:
            assert max_par > 0, "max_par must be greater than 0"
            sema = asyncio.Semaphore(max_par)
            if tqdm:
                import tqdm as tqdm_module

                tqdm_counter = tqdm_module.tqdm(total=len(self))

                async def func_with_semaphore(item: A) -> B:
                    async with sema:
                        result = await func(item)
                        tqdm_counter.update(1)
                        return result

            else:

                async def func_with_semaphore(item: A) -> B:
                    async with sema:
                        return await func(item)

            return Slist(await asyncio.gather(*[func_with_semaphore(item) for item in self]))

    async def gather(self: Sequence[typing.Awaitable[B]]) -> Slist[B]:
        """Gather and await all awaitables in the sequence.

        Returns
        -------
        Slist[B]
            A new Slist containing the awaited results

        Examples
        --------
        >>> async def slow_value(x):
        ...     await asyncio.sleep(0.1)
        ...     return x
        >>> awaitables = [slow_value(1), slow_value(2), slow_value(3)]
        >>> await Slist(awaitables).gather()
        Slist([1, 2, 3])
        """
        return Slist(await asyncio.gather(*self))

    def filter_text_search(self, key: Callable[[A], str], search: List[str]) -> Slist[A]:
        """Filter items based on text search terms.

        Parameters
        ----------
        key : Callable[[A], str]
            Function to extract searchable text from each item
        search : List[str]
            List of search terms to match (case-insensitive)

        Returns
        -------
        Slist[A]
            Items where key text matches any search term

        Examples
        --------
        >>> items = Slist(['apple pie', 'banana bread', 'cherry cake'])
        >>> items.filter_text_search(lambda x: x, ['pie', 'cake'])
        Slist(['apple pie', 'cherry cake'])
        """

        def matches_search(text: str) -> bool:
            if search:
                search_regex = re.compile("|".join(search), re.IGNORECASE)
                return bool(re.search(search_regex, text))
            else:
                return True  # No filter if search undefined

        return self.filter(predicate=lambda item: matches_search(key(item)))

    def mk_string(self: Sequence[str], sep: str) -> str:
        """Join string elements with a separator.

        Parameters
        ----------
        sep : str
            Separator to use between elements

        Returns
        -------
        str
            Joined string

        Examples
        --------
        >>> Slist(['a', 'b', 'c']).mk_string(', ')
        'a, b, c'
        """
        return sep.join(self)

    @overload
    def sum(self: Sequence[int]) -> int: ...

    @overload
    def sum(self: Sequence[float]) -> float: ...

    def sum(
        self: Sequence[Union[int, float, bool]],
    ) -> Union[int, float]:
        """Returns 0 when the list is empty"""
        return sum(self)

    def average(
        self: Sequence[Union[int, float, bool]],
    ) -> Optional[float]:
        """Calculate the arithmetic mean of numeric values.

        Returns
        -------
        Optional[float]
            The average of all values, or None if the list is empty

        Examples
        --------
        >>> Slist([1, 2, 3, 4]).average()
        2.5
        >>> Slist([]).average()
        None
        """
        this = typing.cast(Slist[Union[int, float, bool]], self)
        return this.sum() / this.length if this.length > 0 else None

    def average_or_raise(
        self: Sequence[Union[int, float, bool]],
    ) -> float:
        """Calculate the arithmetic mean of numeric values.

        Returns
        -------
        float
            The average of all values

        Raises
        ------
        ValueError
            If the list is empty

        Examples
        --------
        >>> Slist([1, 2, 3, 4]).average_or_raise()
        2.5
        >>> try:
        ...     Slist([]).average_or_raise()
        ... except ValueError as e:
        ...     print(str(e))
        Cannot get average of empty list
        """
        this = typing.cast(Slist[Union[int, float, bool]], self)
        if this.length == 0:
            raise ValueError("Cannot get average of empty list")
        return this.sum() / this.length

    def statistics_or_raise(
        self: Sequence[Union[int, float, bool]],
    ) -> AverageStats:
        """Calculate comprehensive statistics for numeric values.

        Returns
        -------
        AverageStats
            Statistics including mean, standard deviation, and confidence intervals

        Raises
        ------
        ValueError
            If the list is empty

        Examples
        --------
        >>> stats = Slist([1, 2, 3, 4, 5]).statistics_or_raise()
        >>> round(stats.average, 2)
        3.0
        >>> round(stats.standard_deviation, 2)
        1.58
        """
        this = typing.cast(Slist[Union[int, float, bool]], self)
        if this.length == 0:
            raise ValueError("Cannot get average of empty list")
        average = this.average_or_raise()
        standard_deviation = this.standard_deviation()
        assert standard_deviation is not None
        standard_error = standard_deviation / ((this.length) ** 0.5)
        upper_ci = average + 1.96 * standard_error
        lower_ci = average - 1.96 * standard_error
        average_plus_minus_95 = 1.96 * standard_error
        return AverageStats(
            average=average,
            standard_deviation=standard_deviation,
            upper_confidence_interval_95=upper_ci,
            lower_confidence_interval_95=lower_ci,
            count=this.length,
            average_plus_minus_95=average_plus_minus_95,
        )

    def standard_deviation(self: Slist[Union[int, float]]) -> Optional[float]:
        """Calculate the population standard deviation.

        Returns
        -------
        Optional[float]
            The standard deviation, or None if the list is empty

        Examples
        --------
        >>> round(Slist([1, 2, 3, 4, 5]).standard_deviation(), 2)
        1.58
        >>> Slist([]).standard_deviation()
        None
        """
        return statistics.stdev(self) if self.length > 0 else None

    def standardize(self: Slist[Union[int, float]]) -> Slist[float]:
        """Standardize values to have mean 0 and standard deviation 1.

        Returns
        -------
        Slist[float]
            Standardized values, or empty list if input is empty

        Examples
        --------
        >>> result = Slist([1, 2, 3, 4, 5]).standardize()
        >>> [round(x, 2) for x in result]  # Rounded for display
        [-1.26, -0.63, 0.0, 0.63, 1.26]
        """
        mean = self.average()
        sd = self.standard_deviation()
        return Slist((x - mean) / sd for x in self) if mean is not None and sd is not None else Slist()

    def fold_left(self, acc: B, func: Callable[[B, A], B]) -> B:
        """Fold left operation (reduce) with initial accumulator.

        Parameters
        ----------
        acc : B
            Initial accumulator value
        func : Callable[[B, A], B]
            Function to combine accumulator with each element

        Returns
        -------
        B
            Final accumulated value

        Examples
        --------
        >>> Slist([1, 2, 3, 4]).fold_left(0, lambda acc, x: acc + x)
        10
        >>> Slist(['a', 'b', 'c']).fold_left('', lambda acc, x: acc + x)
        'abc'
        """
        return reduce(func, self, acc)

    def fold_right(self, acc: B, func: Callable[[A, B], B]) -> B:
        """Fold right operation with initial accumulator.

        Parameters
        ----------
        acc : B
            Initial accumulator value
        func : Callable[[A, B], B]
            Function to combine each element with accumulator

        Returns
        -------
        B
            Final accumulated value

        Examples
        --------
        >>> Slist([1, 2, 3]).fold_right('', lambda x, acc: str(x) + acc)
        '321'
        """
        return reduce(lambda a, b: func(b, a), reversed(self), acc)

    def sum_option(self: Sequence[CanAdd]) -> Optional[CanAdd]:
        """Sums the elements of the sequence. Returns None if the sequence is empty.

        Returns
        -------
        Optional[CanAdd]
            The sum of all elements in the sequence, or None if the sequence is empty

        Examples
        --------
        >>> Slist([1, 2, 3]).sum_option()
        6
        >>> Slist([]).sum_option()
        None
        """
        return reduce(lambda a, b: a + b, self) if len(self) > 0 else None

    def sum_or_raise(self: Sequence[CanAdd]) -> CanAdd:
        """Sums the elements of the sequence. Raises an error if the sequence is empty.

        Returns
        -------
        CanAdd
            The sum of all elements in the sequence

        Raises
        ------
        AssertionError
            If the sequence is empty

        Examples
        --------
        >>> Slist([1, 2, 3]).sum_or_raise()
        6
        >>> Slist([]).sum_or_raise()  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        AssertionError: Cannot fold empty list
        """
        assert len(self) > 0, "Cannot fold empty list"
        return reduce(lambda a, b: a + b, self)

    def split_by(self, predicate: Callable[[A], bool]) -> Tuple[Slist[A], Slist[A]]:
        """Split list into two lists based on a predicate. Left list contains items that match the predicate.

        Parameters
        ----------
        predicate : Callable[[A], bool]
            Function to determine which list each element goes into

        Returns
        -------
        Tuple[Slist[A], Slist[A]]
            Tuple of (matching elements, non-matching elements)

        Examples
        --------
        >>> evens, odds = Slist([1, 2, 3, 4, 5]).split_by(lambda x: x % 2 == 0)
        >>> evens
        Slist([2, 4])
        >>> odds
        Slist([1, 3, 5])
        """
        left = Slist[A]()
        right = Slist[A]()
        for item in self:
            if predicate(item):
                left.append(item)
            else:
                right.append(item)
        return left, right

    def split_on(self, predicate: Callable[[A], bool]) -> Slist[Slist[A]]:
        """Split list into sublists based on a predicate.

        Returns
        -------
        Slist[Slist[A]]
            List of sublists

        Examples
        --------
        >>> Slist([1, 2, 3, 4, 5]).split_on(lambda x: x % 2 == 0)
        Slist([Slist([1, 3, 5]), Slist([2, 4])])
        """
        output: Slist[Slist[A]] = Slist()
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
        """Split list into two parts based on a proportion.

        Parameters
        ----------
        left_proportion : float
            Proportion of elements to include in first list (0 < left_proportion < 1)

        Returns
        -------
        Tuple[Slist[A], Slist[A]]
            Tuple of (first part, second part)

        Examples
        --------
        >>> first, second = Slist([1, 2, 3, 4, 5]).split_proportion(0.6)
        >>> first
        Slist([1, 2, 3])
        >>> second
        Slist([4, 5])
        """
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
        """Split list into n roughly equal parts.

        Parameters
        ----------
        n : int
            Number of parts to split into (must be positive)

        Returns
        -------
        Slist[Slist[A]]
            List of n sublists of roughly equal size

        Examples
        --------
        >>> Slist([1, 2, 3, 4, 5]).split_into_n(2)
        Slist([Slist([1, 3, 5]), Slist([2, 4])])
        """
        assert n > 0, "n needs to be greater than 0"
        output: Slist[Slist[A]] = Slist()
        for _ in range(n):
            output.append(Slist[A]())
        for idx, item in enumerate(self):
            output[idx % n].append(item)
        return output

    def copy(self) -> Slist[A]:
        """Create a shallow copy of the list.

        Returns
        -------
        Slist[A]
            A new Slist with the same elements

        Examples
        --------
        >>> original = Slist([1, 2, 3])
        >>> copied = original.copy()
        >>> copied.append(4)
        >>> original  # Original is unchanged
        Slist([1, 2, 3])
        """
        return Slist(super().copy())

    def repeat_until_size(self, size: int) -> Optional[Slist[A]]:
        """Repeat the list elements until reaching specified size.

        Parameters
        ----------
        size : int
            Target size (must be positive)

        Returns
        -------
        Optional[Slist[A]]
            New list with repeated elements, or None if input is empty

        Examples
        --------
        >>> Slist([1, 2]).repeat_until_size(5)
        Slist([1, 2, 1, 2, 1])
        >>> Slist([]).repeat_until_size(3)
        None
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

    def repeat_until_size_enumerate(self, size: int) -> Slist[Tuple[int, A]]:
        """Repeat the list elements until reaching specified size, with enumeration.

        Parameters
        ----------
        size : int
            Target size (must be positive)

        Returns
        -------
        Slist[Tuple[int, A]]
            New list with repeated elements and their repetition count

        Raises
        ------
        AssertionError
            If size is not positive
        ValueError
            If input list is empty

        Examples
        --------
        >>> Slist(["a", "b"]).repeat_until_size_enumerate(5)
        Slist([(0, 'a'), (0, 'b'), (1, 'a'), (1, 'b'), (2, 'a')])
        """
        assert size > 0, "size needs to be greater than 0"
        if self.is_empty:
            raise ValueError("input needs to be non empty")

        new = Slist[Tuple[int, A]]()
        repetition_count = 0

        while True:
            for item in self:
                if len(new) >= size:
                    return new
                else:
                    new.append((repetition_count, item))
            repetition_count += 1

    def repeat_until_size_or_raise(self, size: int) -> Slist[A]:
        """Repeat the list elements until reaching specified size.

        Parameters
        ----------
        size : int
            Target size (must be positive)

        Returns
        -------
        Slist[A]
            New list with repeated elements

        Raises
        ------
        AssertionError
            If size is not positive
        ValueError
            If input list is empty

        Examples
        --------
        >>> Slist([1, 2]).repeat_until_size_or_raise(5)
        Slist([1, 2, 1, 2, 1])
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

    @overload
    def zip(self, other: Sequence[B], /) -> Slist[Tuple[A, B]]: ...

    @overload
    def zip(self, other1: Sequence[B], other2: Sequence[C], /) -> Slist[Tuple[A, B, C]]: ...

    @overload
    def zip(self, other1: Sequence[B], other2: Sequence[C], other3: Sequence[D], /) -> Slist[Tuple[A, B, C, D]]: ...

    @overload
    def zip(
        self, other1: Sequence[B], other2: Sequence[C], other3: Sequence[D], other4: Sequence[E], /
    ) -> Slist[Tuple[A, B, C, D, E]]: ...

    def zip(self: Sequence[A], *others: Sequence[Any]) -> Slist[Tuple[Any, ...]]:
        """Zip this list with other sequences.

        Parameters
        ----------
        *others : Sequence[B]
            Other sequences to zip with

        Returns
        -------
        Slist[Tuple[A, *Tuple[B, ...]]]
            List of tuples containing elements from all sequences

        Raises
        ------
        TypeError
            If sequences have different lengths

        Examples
        --------
        >>> Slist([1, 2, 3]).zip(Slist(["1", "2", "3"]))
        Slist([(1, "1"), (2, "2"), (3, "3")])
        >>> Slist([1, 2, 3]).zip(Slist(["1", "2", "3"]), Slist([True, True, True]))
        Slist([(1, "1", True), (2, "2", True), (3, "3", True)])
        """
        # Convert to list to check lengths
        if sys.version_info >= (3, 10):
            return Slist(zip(self, *others, strict=True))
        else:
            return Slist(zip(self, *others))

    @overload
    def zip_cycle(self, other: Sequence[B], /) -> Slist[Tuple[A, B]]: ...

    @overload
    def zip_cycle(self, other1: Sequence[B], other2: Sequence[C], /) -> Slist[Tuple[A, B, C]]: ...

    @overload
    def zip_cycle(
        self, other1: Sequence[B], other2: Sequence[C], other3: Sequence[D], /
    ) -> Slist[Tuple[A, B, C, D]]: ...

    @overload
    def zip_cycle(
        self, other1: Sequence[B], other2: Sequence[C], other3: Sequence[D], other4: Sequence[E], /
    ) -> Slist[Tuple[A, B, C, D, E]]: ...

    def zip_cycle(self: Sequence[A], *others: Sequence[Any]) -> Slist[Tuple[Any, ...]]:
        """Zip sequences by cycling shorter ones until all are exhausted.

        Unlike regular zip which stops at the shortest sequence, zip_cycle
        repeats shorter sequences cyclically until the longest sequence is exhausted.

        Parameters
        ----------
        *others : Sequence[Any]
            Other sequences to zip with

        Returns
        -------
        Slist[Tuple[Any, ...]]
            List of tuples containing elements from all sequences

        Examples
        --------
        >>> Slist([1, 2, 3]).zip_cycle(['a', 'b'])
        Slist([(1, 'a'), (2, 'b'), (3, 'a')])
        >>> Slist([1, 2]).zip_cycle(['a', 'b', 'c', 'd'])
        Slist([(1, 'a'), (2, 'b'), (1, 'c'), (2, 'd')])
        >>> Slist([1, 2, 3]).zip_cycle(['a', 'b'], [10, 20, 30, 40])
        Slist([(1, 'a', 10), (2, 'b', 20), (3, 'a', 30), (1, 'b', 40)])
        """
        all_sequences = [self] + list(others)
        
        # If any sequence is empty, return empty list
        if any(len(seq) == 0 for seq in all_sequences):
            return Slist()
        
        # Find the maximum length
        max_len = max(len(seq) for seq in all_sequences)
        
        # Create cycled iterators for each sequence
        cycled_iterators = [itertools.cycle(seq) for seq in all_sequences]
        
        # Zip them together for max_len items
        result = []
        for _ in range(max_len):
            result.append(tuple(next(it) for it in cycled_iterators))
        
        return Slist(result)

    def slice_with_bool(self, bools: Sequence[bool]) -> Slist[A]:
        """Slice the list using a sequence of boolean values.

        Parameters
        ----------
        bools : Sequence[bool]
            Boolean sequence indicating which elements to keep

        Returns
        -------
        Slist[A]
            List containing elements where corresponding boolean is True

        Examples
        --------
        >>> Slist([1, 2, 3, 4, 5]).slice_with_bool([True, False, True, False, True])
        Slist([1, 3, 5])
        """
        return Slist(item for item, keep in zip(self, bools) if keep)

    def __mul__(self, other: typing.SupportsIndex) -> Slist[A]:
        return Slist(super().__mul__(other))

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler):  # type: ignore
        # only called by pydantic v2
        from pydantic_core import core_schema  # type: ignore

        return core_schema.no_info_after_validator_function(cls, handler(list))

    def find_one_or_raise(
        self,
        predicate: Callable[[A], bool],
        exception: Exception = RuntimeError("Failed to find predicate"),
    ) -> A:
        """Find first element that satisfies a predicate or raise exception.

        Parameters
        ----------
        predicate : Callable[[A], bool]
            Function that returns True for the desired element
        exception : Exception, optional
            Exception to raise if no match found, by default RuntimeError("Failed to find predicate")

        Returns
        -------
        A
            First matching element

        Raises
        ------
        Exception
            If no matching element is found

        Examples
        --------
        >>> Slist([1, 2, 3, 4]).find_one_or_raise(lambda x: x > 3)
        4
        >>> try:
        ...     Slist([1, 2, 3]).find_one_or_raise(lambda x: x > 5)
        ... except RuntimeError as e:
        ...     print(str(e))
        Failed to find predicate
        """
        result = self.find_one(predicate)
        if result is not None:
            return result
        else:
            raise exception

    def permutations_pairs(self) -> Slist[Tuple[A, A]]:
        """Generate all possible pairs of elements, including reversed pairs.

        This method uses itertools.permutations with length=2,
        but filters out pairs where both elements are the same.

        Returns
        -------
        Slist[Tuple[A, A]]
            A new Slist containing all pairs of elements

        Examples
        --------
        >>> Slist([1, 2]).permutations_pairs()
        Slist([(1, 2), (2, 1)])
        >>> Slist([1, 2, 3]).permutations_pairs()
        Slist([(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)])
        >>> Slist([]).permutations_pairs()
        Slist([])
        >>> Slist([1]).permutations_pairs()
        Slist([])
        """
        result = Slist(perm for perm in itertools.permutations(self, 2))
        return result

    def combinations_pairs(self) -> Slist[Tuple[A, A]]:
        """Generate pairs of elements without including reversed pairs.

        This method uses itertools.combinations with length=2.

        Returns
        -------
        Slist[Tuple[A, A]]
            A new Slist containing unique pairs of elements

        Examples
        --------
        >>> Slist([1, 2]).combinations_pairs()
        Slist([(1, 2)])
        >>> Slist([1, 2, 3]).combinations_pairs()
        Slist([(1, 2), (1, 3), (2, 3)])
        >>> Slist([]).combinations_pairs()
        Slist([])
        >>> Slist([1]).combinations_pairs()
        Slist([])
        """
        result = Slist(itertools.combinations(self, 2))
        return result
