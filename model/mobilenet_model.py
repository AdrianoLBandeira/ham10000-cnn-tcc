"""
Criação e ajuste fino do modelo CNN baseado em MobileNet.
"""
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall

from utils.config import INITIAL_LEARNING_RATE, FINE_TUNE_LEARNING_RATE, DROPOUT_RATE


def _compile_model(model, learning_rate):
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", Precision(name="precision"), Recall(name="recall")],
    )
    return model


def create_model(num_classes, image_size=(224, 224), dropout_rate=DROPOUT_RATE):
    """
    Cria um modelo de classificação baseado na MobileNet pré-treinada na ImageNet.

    A base convolucional é congelada inicialmente. Apenas as camadas adicionadas ao topo
    são treinadas na primeira etapa.
    """

    input_shape = (image_size[0], image_size[1], 3)
    inputs = Input(shape=input_shape)

    base_model = MobileNet(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
        name="mobilenet_base"
    )

    # Mantém um nome previsível para permitir fine-tuning via model.get_layer().
    try:
        base_model._name = "mobilenet_base"
    except AttributeError:
        pass

    base_model.trainable = False

    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="ham10000_mobilenet")
    return _compile_model(model, INITIAL_LEARNING_RATE)


def fine_tune_model(model, layers_to_unfreeze=25):
    """
    Descongela as últimas camadas da base MobileNet para realizar fine-tuning.
    """

    try:
        base_model = model.get_layer("mobilenet_base")
    except ValueError:
        base_model = model.get_layer("mobilenet_1.00_224")

    base_model.trainable = True

    for layer in base_model.layers[:-layers_to_unfreeze]:
        layer.trainable = False

    for layer in base_model.layers[-layers_to_unfreeze:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model