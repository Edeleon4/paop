import paop
import unittest

class TestSVM(unittest.TestCase):
    def test_init_bow(self):
        dmy_data = "tests/dmy_data.p"
        lstm = paop.LSTMMergeClassifyBOW(dmy_data, dmy_data, dmy_data)
        self.assertIsNotNone(lstm)
    def test_train_bow(self):
        dmy_data = "tests/dmy_data.p"
        lstm = paop.LSTMMergeClassifyBOW(dmy_data, dmy_data, dmy_data)
        lstm.train()
    def test_eval_bow(self):
        dmy_data = "tests/dmy_data.p"
        lstm = paop.LSTMMergeClassifyBOW(dmy_data, dmy_data, dmy_data)
        lstm.train()
        score_out = lstm.evaluate()
        self.assertTrue(score_out[1]< .40)

    def test_init_embeds(self):
        dmy_data = "tests/dmy_data.p"
        lstm = paop.LSTMMergeClassifyEmbeds(dmy_data, dmy_data, dmy_data)
        self.assertIsNotNone(lstm)
    def test_train_embeds(self):
        dmy_data = "tests/dmy_data.p"
        lstm = paop.LSTMMergeClassifyEmbeds(dmy_data, dmy_data, dmy_data)
        lstm.train()
    def test_eval_embeds(self):
        dmy_data = "tests/dmy_data.p"
        lstm = paop.LSTMMergeClassifyEmbeds(dmy_data, dmy_data, dmy_data)
        lstm.train()
        score_out = lstm.evaluate()
        self.assertTrue(score_out[1]< .40)

if __name__ == '__main__':
    unittest.main()
