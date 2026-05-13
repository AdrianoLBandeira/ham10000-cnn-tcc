"""
Download do dataset HAM10000 via KaggleHub.
"""

import os
import kagglehub

from utils.config import DATASET_NAME


def download_dataset(dataset_name=DATASET_NAME):
    """
    Baixa o dataset informado no KaggleHub e retorna o caminho local.

    O KaggleHub mantém cache local. Portanto, após o primeiro download,
    as próximas execuções tendem a reutilizar os arquivos já baixados.
    """

    dataset_path = kagglehub.dataset_download(dataset_name)

    required_files = [
        "HAM10000_metadata.csv",
        "HAM10000_images_part_1",
        "HAM10000_images_part_2",
    ]

    missing = [
        item for item in required_files
        if not os.path.exists(os.path.join(dataset_path, item))
    ]

    if missing:
        raise FileNotFoundError(
            "O dataset foi baixado, mas alguns itens esperados não foram encontrados: "
            + ", ".join(missing)
        )

    print(f"Dataset disponível em: {dataset_path}")
    return dataset_path
