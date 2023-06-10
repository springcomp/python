from tests import unittest

class TestSample(unittest.TestCase):
	def setUp(self):
		pass

	def test_x(self):
		self.assertEqual(42, 42)

if __name__ == '__main__':
	unittest.main()