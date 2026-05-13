"""
Cálculo de pesos das classes para tratar o desbalanceamento do HAM10000.
"""

import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def calculate_class_weights(train_generator):
    """
    Calcula pesos para cada classe com base na frequência das amostras de treinamento.

    Classes minoritárias recebem peso maior durante o treinamento, reduzindo a tendência
    de o modelo privilegiar a classe majoritária.
    """

    class_labels = train_generator.classes

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(class_labels),
        y=class_labels,
    )

    class_weights_dict = dict(enumerate(class_weights))

    print("\nPesos calculados para as classes:")
    for class_index, weight in class_weights_dict.items():
        print(f"Classe {class_index}: {weight:.4f}")

    return class_weights_dict
