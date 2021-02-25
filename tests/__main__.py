"""
To be ran from the root folder with:

coverage run -m tests
coverage html
"""

import unittest

loader = unittest.TestLoader()
suite = loader.discover('tests', pattern='test_*.py')

runner = unittest.TextTestRunner()
runner.run(suite)


if __name__ == '__main__':
    pass
