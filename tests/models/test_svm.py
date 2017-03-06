import paop
import unittest

class TestSVM(unittest.TestCase):
    def test_init(self):
        dmy_data = "tests/dmy_data.json"
        svm = paop.SVM(dmy_data, dmy_data, dmy_data)
        self.assertIsNotNone(dmy_data)

if __name__ == '__main__':
    unittest.main()
