"""Shared pytest options for tests requiring a real repository directory.

pytest_addoption must live in conftest.py to be recognized before command-line
parsing; defining it directly in test modules (as test_caching.py does) has no
effect on its own.
"""


def pytest_addoption(parser):
    parser.addoption("--repo", action="store", default=None, help="Path to repo dir")
