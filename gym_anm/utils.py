"""Utility functions."""

import os


def get_package_root():
    """Return absolute path to the package root."""
    return os.path.dirname(os.path.abspath(__file__))
