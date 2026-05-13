"""
Carregamento dos metadados do HAM10000 e associação com os caminhos das imagens.
"""

import os
import pandas as pd


CLASS_NAMES_PT = {
    "akiec": "Ceratose actínica / carcinoma intraepitelial",
    "bcc": "Carcinoma basocelular",
    "bkl": "Lesão queratósica benigna",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Nevo melanocítico",
    "vasc": "Lesão vascular",
}


def load_metadata(dataset_path):
    """
    Lê o arquivo HAM10000_metadata.csv e cria a coluna 'path' com o caminho completo
    de cada imagem.
    """

    metadata_file = os.path.join(dataset_path, "HAM10000_metadata.csv")

    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Arquivo de metadados não encontrado: {metadata_file}")

    df = pd.read_csv(metadata_file)

    image_dirs = [
        os.path.join(dataset_path, "HAM10000_images_part_1"),
        os.path.join(dataset_path, "HAM10000_images_part_2"),
    ]

    image_paths = {}

    for directory in image_dirs:
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Diretório de imagens não encontrado: {directory}")

        for filename in os.listdir(directory):
            if filename.lower().endswith(".jpg"):
                image_id = os.path.splitext(filename)[0]
                image_paths[image_id] = os.path.join(directory, filename)

    df["path"] = df["image_id"].map(image_paths.get)
    df["dx_label_pt"] = df["dx"].map(CLASS_NAMES_PT)

    before_drop = len(df)
    df = df.dropna(subset=["path", "dx"]).reset_index(drop=True)
    removed = before_drop - len(df)

    print(f"Total de registros válidos: {len(df)}")
    if removed:
        print(f"Registros removidos por falta de imagem/rótulo: {removed}")

    print("\nDistribuição das classes:")
    print(df["dx"].value_counts())

    return df
