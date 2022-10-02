import os
import pandas

def read(file_path: str, delimiter: str) -> pandas.DataFrame:
    dataset = pandas.read_csv(config.file_path, delimiter=delimiter)
    return dataset
