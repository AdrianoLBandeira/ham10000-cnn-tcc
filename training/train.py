"""
Criação dos geradores de imagens e treinamento da rede neural.
"""

import os

import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils.config import FIGURE_DIR, MODEL_DIR, RANDOM_STATE


def create_generators(train_df, validation_df, test_df, image_size, batch_size):
    """
    Cria geradores para treino, validação e teste.

    O aumento de dados é aplicado apenas no treinamento. Validação e teste recebem apenas
    o pré-processamento exigido pela MobileNet.
    """

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest",
    )

    eval_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    )

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="path",
        y_col="dx",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=RANDOM_STATE,
    )

    validation_generator = eval_datagen.flow_from_dataframe(
        dataframe=validation_df,
        x_col="path",
        y_col="dx",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    test_generator = eval_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="path",
        y_col="dx",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    return train_generator, validation_generator, test_generator


def _create_callbacks(phase_name):
    os.makedirs(MODEL_DIR, exist_ok=True)

    return [
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, f"best_model_{phase_name}.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        CSVLogger(os.path.join(MODEL_DIR, f"training_log_{phase_name}.csv")),
    ]


def train_model(model, train_generator, validation_generator, epochs, class_weights=None, phase_name="initial"):
    """
    Treina o modelo e retorna o histórico de treinamento.
    """

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=_create_callbacks(phase_name),
    )

    return history


def save_training_curves(history, filename_prefix):
    """
    Salva gráficos de acurácia e perda para posterior uso no Capítulo 4.
    """

    os.makedirs(FIGURE_DIR, exist_ok=True)

    plt.figure()
    plt.plot(history.history.get("accuracy", []), label="Treino")
    plt.plot(history.history.get("val_accuracy", []), label="Validação")
    plt.title("Acurácia por época")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.savefig(os.path.join(FIGURE_DIR, f"{filename_prefix}_accuracy.png"), bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(history.history.get("loss", []), label="Treino")
    plt.plot(history.history.get("val_loss", []), label="Validação")
    plt.title("Perda por época")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(FIGURE_DIR, f"{filename_prefix}_loss.png"), bbox_inches="tight")
    plt.close()
