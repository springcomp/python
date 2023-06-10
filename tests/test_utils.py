from tests import unittest
from experiment.utils import get_max_length_arrays
from experiment.utils import pad_arrays_to_length


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.results = {
            'one': [1],
            'two': [1, 2],
            'three': [1, 2, 3],
            'four': [1, 2, 3, 4]
        }

    def test_get_max_length_arrays(self):
        self.assertEqual(4, get_max_length_arrays(self.results))

    def test_pad_arrays_to_length(self):
        actual = self.results.copy()
        expected = {
            'one': [1, None, None, None],
            'two': [1, 2, None, None],
            'three': [1, 2, 3, None],
            'four': [1, 2, 3, 4]
        }
        pad_arrays_to_length(actual, get_max_length_arrays(actual))
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
