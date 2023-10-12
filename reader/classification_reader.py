from .base_reader import DataReader
from loguru import logger
from glob import glob
from torch.utils.data import DataLoader, Dataset
from configs.data_config import DataConfig


class ClassificationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"text": self.data[idx]["text"], "labels": self.data[idx]["label"]}


class ClassificationReader(DataReader):
    def __init__(self, config: DataConfig, exp_id: str):
        super().__init__(config, exp_id)
        self.exp_id = exp_id

    def read(self, batch_size, eval_flag=False):
        if eval_flag:
            df = self.test_df
        else:
            df = self.train_df

        dataset = ClassificationDataset(df)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return dataloader
