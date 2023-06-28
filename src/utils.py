import json

from datasets import Dataset, DatasetDict


def read_dataset(file_path: str) -> DatasetDict:
    """
    file_path: str
        Path to the dataset file
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    dataset = DatasetDict({
        "dataset": Dataset.from_list(data["dataset"])
    })

    return dataset
