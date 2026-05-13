from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def create_model(num_classes):

    base_model = MobileNet(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model._name = "mobilenet_base"

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)

    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)

    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)

    output = Dense(num_classes, activation="softmax")(x)

    model = Model(
        inputs=base_model.input,
        outputs=output
    )

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model