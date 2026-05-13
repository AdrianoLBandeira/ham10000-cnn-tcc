"""
Download do dataset HAM10000 utilizando KaggleHub
"""

import kagglehub
from utils.config import DATASET_NAME

def download_dataset(dataset_name=DATASET_NAME):

    print("Baixando dataset HAM10000 via KaggleHub...")

    dataset_path = kagglehub.dataset_download(dataset_name)

    print("Dataset baixado para:")

    print(dataset_path)

    return dataset_path