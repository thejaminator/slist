# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at https://github.com/thejaminator/slist/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

slist could always use more documentation, whether as part of the
official slist docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at https://github.com/thejaminator/slist/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `slist` for local development.

1. Fork the `slist` repo on GitHub.
2. Clone your fork locally

    ```
    $ git clone git@github.com:your_name_here/slist.git
    ```

3. Ensure [poetry](https://python-poetry.org/docs/) is installed.
4. Install dependencies and start your virtualenv:

    ```
    $ poetry install -E test -E doc -E dev
    $ pip install tox
    ```

5. Create a branch for local development:

    ```
    $ git checkout -b name-of-your-bugfix-or-feature
    ```

    Now you can make your changes locally.

6. When you're done making changes, check that your changes pass the
   tests, including testing other Python versions, with tox:

    ```
    $ poetry run pytest tests
    ```

7. Commit your changes and push your branch to GitHub:

    ```
    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature
    ```

8. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.md.
3. The pull request should work for Python 3.6, 3.7, 3.8 and 3.9. Check
   https://github.com/thejaminator/slist/actions
   and make sure that the tests pass for all supported Python versions.

## Tips

```
$ poetry run pytest tests/test_slist.py
```

To run a subset of tests.


## Deploying

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in CHANGELOG.md).
Then run:

```
$ poetry run bump2version patch # possible: major / minor / patch
$ git push
$ git push --tags
```

GitHub Actions will then deploy to PyPI if tests pass.

## Documentation Guidelines

### Docstring Format
All new code should include docstrings following the reST/NumPy format:

```python
def method_name(self, param1: Type1, param2: Type2) -> ReturnType:
    """Short description of what the method does.

    Parameters
    ----------
    param1 : Type1
        Description of param1
    param2 : Type2
        Description of param2

    Returns
    -------
    ReturnType
        Description of return value

    Examples
    --------
    >>> Slist([1, 2, 3]).method_name(param1, param2)
    Expected output
    """
```

### Documentation Requirements

1. All public methods must have docstrings
2. Include type information in Parameters and Returns sections
3. Provide at least one working example
4. Use backticks (`` ` ``) for inline code references
5. Keep examples simple and focused on one use case
6. Include edge cases in examples where relevant

### Building Documentation

1. Install documentation dependencies:
```bash
poetry install -E doc
```

2. Preview documentation locally:
```bash
mkdocs serve
```

3. Build documentation:
```bash
mkdocs build
```

The documentation will automatically be built and deployed when changes are merged to main.

### Documentation Structure

- `docs/index.md`: Main landing page and quick start
- `docs/api/`: API reference documentation
- `docs/contributing.md`: Contribution guidelines (this file)

### Tips for Good Documentation

1. Write clear, concise descriptions
2. Include both basic and advanced examples
3. Document exceptions and edge cases
4. Keep examples runnable and tested
5. Update docs when changing method signatures
6. Use proper formatting for code blocks and inline code
