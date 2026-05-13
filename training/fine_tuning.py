from tensorflow.keras.optimizers import Adam


def fine_tune_model(model, layers_to_unfreeze=25):

    base_model = model.get_layer("mobilenet_base")

    for layer in base_model.layers:
        layer.trainable = False

    for layer in base_model.layers[-layers_to_unfreeze:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model