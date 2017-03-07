import paop
import unittest

class TestSVM(unittest.TestCase):
    def test_init(self):
        dmy_data = "tests/dmy_data.json"
        svm = paop.SVM(dmy_data, dmy_data, dmy_data)
        self.assertIsNotNone(dmy_data)
    def test_train(self):
        dmy_data = "tests/dmy_data.json"
        svm = paop.SVM(dmy_data, dmy_data, dmy_data)
        svm.train()
    def test_eval(self):
        dmy_data = "tests/dmy_data.json"
        svm = paop.SVM(dmy_data, dmy_data, dmy_data)
        svm.train()
        score_out = model.evaluate()
        self.assertTrue(score_out > .9)


if __name__ == '__main__':
    unittest.main()
