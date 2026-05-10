"""
Cálculo de pesos das classes para tratar desbalanceamento do HAM10000.
"""

import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def calculate_class_weights(train_generator):
    """
    Calcula pesos para cada classe com base na frequência das amostras.

    Classes com menos imagens recebem peso maior.
    Classes com mais imagens recebem peso menor.
    """

    class_labels = train_generator.classes

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(class_labels),
        y=class_labels
    )

    class_weights_dict = dict(enumerate(class_weights))

    print("\nPesos calculados para as classes:")
    print(class_weights_dict)

    return class_weights_dict