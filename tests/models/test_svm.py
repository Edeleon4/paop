import paop
import unittest

class TestSVM(unittest.TestCase):
    def test_init(self):
        svm = paop.SVM("dmy_data.json", "dmy_data.json", "dmy_data.json")
        self.assertIsNotNone(b.board)

