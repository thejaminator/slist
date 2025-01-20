# Welcome to Slist Documentation

Slist is a typesafe list implementation for Python that provides enhanced method chaining capabilities and additional utility functions.

## Features

- 🔒 Type-safe: Full type hints and mypy support
- ⛓️ Method chaining: Fluent interface for list operations
- 🛠️ Rich functionality: Many utility methods for common operations
- 🚀 Performance: Minimal overhead over Python's built-in list
- 🔍 Clear API: Well-documented methods with intuitive names

## Installation

```bash
pip install slist
```

Or with Poetry:

```bash
poetry add slist
```

## Quick Example

```python
from slist import Slist

# Create a list of numbers
numbers = Slist([1, 2, 3, 4, 5])

# Chain operations
result = numbers\
    .filter(lambda x: x % 2 == 0)\  # Keep even numbers
    .map(lambda x: x * 2)\          # Double each number
    .reversed()\                    # Reverse the order
    .add_one(10)                    # Add one more number

print(result)  # Slist([10, 8, 4])
```

## Why Slist?

Slist enhances Python's built-in list with:

1. Method chaining for cleaner code
2. Type-safe operations
3. Additional utility methods
4. Functional programming patterns
5. Async operation support

Check out the [API Reference](api/slist.md) for detailed documentation of all available methods. 