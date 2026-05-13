"""
Cálculo de pesos das classes para tratar desbalanceamento do HAM10000.
"""

import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def calculate_class_weights(train_generator, smooth=True):

    class_labels = train_generator.classes

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(class_labels),
        y=class_labels
    )

    if smooth:
        class_weights = np.sqrt(class_weights)

    class_weights_dict = dict(enumerate(class_weights))

    print("\nPesos calculados para as classes:")
    print(class_weights_dict)

    return class_weights_dict