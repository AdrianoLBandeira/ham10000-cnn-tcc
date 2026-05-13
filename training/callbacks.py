"""
Callbacks para melhorar o treinamento.
"""

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau


def get_callbacks():

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),

        ModelCheckpoint(
            filepath="best_mobilenet_ham10000.keras",
            monitor="val_loss",
            save_best_only=True
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-7
        )
    ]

    return callbacks