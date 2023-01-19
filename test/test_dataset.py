import unittest

from dataset import TranslateDataModule


class Config:
    data_path = "../../multi30k-dataset/data/"
    batch_size = 8


class TestDataset(unittest.TestCase):
    def test_load(self):
        test_config = Config()
        datamodule = TranslateDataModule(test_config)
        self.assertIsInstance(datamodule, TranslateDataModule)


if __name__ == '__main__':
    unittest.main()
