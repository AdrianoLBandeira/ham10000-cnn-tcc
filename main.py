"""
Pipeline completo do projeto
"""

from dataset.download_dataset import download_dataset
from data.data_loader import load_metadata
from data.data_preprocessing import split_dataset
from model.mobilenet_model import create_model
from training.train import create_generators, train_model
from evaluation.metrics import evaluate_model
from utils.config import IMAGE_SIZE, BATCH_SIZE, EPOCHS, NUM_CLASSES


def main():

    dataset_path = download_dataset()

    df = load_metadata(dataset_path)

    train_df, test_df = split_dataset(df)

    train_generator, test_generator = create_generators(
        train_df,
        test_df,
        IMAGE_SIZE,
        BATCH_SIZE
    )

    model = create_model(NUM_CLASSES)

    train_model(
        model,
        train_generator,
        test_generator,
        EPOCHS
    )

    evaluate_model(
        model,
        test_generator
    )


if __name__ == "__main__":
    main()