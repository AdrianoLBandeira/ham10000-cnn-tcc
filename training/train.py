"""
Treinamento da rede neural
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input


def create_generators(train_df, test_df, image_size, batch_size):

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest"
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="path",
        y_col="dx",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="path",
        y_col="dx",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    return train_generator, test_generator


def train_model(
    model,
    train_generator,
    test_generator,
    epochs,
    class_weights=None,
    callbacks=None
):

    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )

    return history