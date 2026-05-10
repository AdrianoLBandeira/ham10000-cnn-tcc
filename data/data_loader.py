"""
Carrega metadados e cria caminhos das imagens
"""

import os
import pandas as pd


def load_metadata(dataset_path):

    metadata_file = os.path.join(
        dataset_path,
        "HAM10000_metadata.csv"
    )

    df = pd.read_csv(metadata_file)

    image_dir1 = os.path.join(
        dataset_path,
        "HAM10000_images_part_1"
    )

    image_dir2 = os.path.join(
        dataset_path,
        "HAM10000_images_part_2"
    )

    image_paths = {}

    for directory in [image_dir1, image_dir2]:

        for file in os.listdir(directory):

            if file.endswith(".jpg"):

                image_id = file.replace(".jpg", "")

                image_paths[image_id] = os.path.join(
                    directory,
                    file
                )

    df["path"] = df["image_id"].map(image_paths.get)

    print("Total de imagens:", len(df))

    return df