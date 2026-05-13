"""
Pipeline completo do projeto
"""

from dataset.download_dataset import download_dataset

from data.data_loader import load_metadata
from data.data_preprocessing import split_dataset
from data.class_balance import calculate_class_weights

from model.mobilenet_model import create_model

from training.train import create_generators
from training.train import train_model
from training.callbacks import get_callbacks
from training.fine_tuning import fine_tune_model

from evaluation.metrics import evaluate_model

from utils.config import IMAGE_SIZE
from utils.config import BATCH_SIZE
from utils.config import EPOCHS
from utils.config import NUM_CLASSES


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

    class_weights = calculate_class_weights(train_generator)

    callbacks = get_callbacks()

    model = create_model(NUM_CLASSES)

    print("\nEtapa 1: treinamento da cabeça classificadora\n")

    train_model(
        model=model,
        train_generator=train_generator,
        test_generator=test_generator,
        epochs=EPOCHS,
        class_weights=class_weights,
        callbacks=callbacks
    )

    print("\nEtapa 2: fine tuning da MobileNet\n")

    model = fine_tune_model(
        model,
        layers_to_unfreeze=25
    )

    train_model(
        model=model,
        train_generator=train_generator,
        test_generator=test_generator,
        epochs=10,
        class_weights=class_weights,
        callbacks=callbacks
    )

    evaluate_model(
        model,
        test_generator
    )


if __name__ == "__main__":
    main()