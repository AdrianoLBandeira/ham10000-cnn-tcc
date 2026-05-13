"""
Pipeline completo do projeto de classificação de lesões de pele com HAM10000 e MobileNet.
"""

from dataset.download_dataset import download_dataset

from data.class_balance import calculate_class_weights
from data.data_loader import load_metadata
from data.data_preprocessing import split_dataset

from evaluation.metrics import evaluate_model

from model.mobilenet_model import create_model, fine_tune_model

from training.train import create_generators, save_training_curves, train_model

from utils.config import (
    BATCH_SIZE,
    EPOCHS,
    FINE_TUNE_EPOCHS,
    IMAGE_SIZE,
    LAYERS_TO_UNFREEZE,
    NUM_CLASSES,
)


def main():
    dataset_path = download_dataset()

    df = load_metadata(dataset_path)

    train_df, validation_df, test_df = split_dataset(df)

    train_generator, validation_generator, test_generator = create_generators(
        train_df,
        validation_df,
        test_df,
        IMAGE_SIZE,
        BATCH_SIZE,
    )

    class_weights = calculate_class_weights(train_generator)

    model = create_model(NUM_CLASSES, IMAGE_SIZE)

    initial_history = train_model(
        model,
        train_generator,
        validation_generator,
        EPOCHS,
        class_weights,
        phase_name="initial",
    )
    save_training_curves(initial_history, "initial")

    model = fine_tune_model(model, layers_to_unfreeze=LAYERS_TO_UNFREEZE)

    fine_tune_history = train_model(
        model,
        train_generator,
        validation_generator,
        FINE_TUNE_EPOCHS,
        class_weights,
        phase_name="fine_tune",
    )
    save_training_curves(fine_tune_history, "fine_tune")

    evaluate_model(model, test_generator)


if __name__ == "__main__":
    main()
