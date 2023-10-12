from pydantic import BaseModel


class DataSplitConfig(BaseModel):
    '''Base configuration for specifying data splitting parameters'''
    train_ratio:  float
    test_ratio:  float


class DataConfig(BaseModel):
    '''Base configuration for specifying input data parameters'''
    input:  str
    test_split: DataSplitConfig
