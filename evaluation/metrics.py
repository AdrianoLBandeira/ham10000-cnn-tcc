"""
Cálculo das métricas
"""

import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def evaluate_model(model, generator):

    predictions = model.predict(generator)

    y_pred = np.argmax(predictions, axis=1)

    y_true = generator.classes

    print("\nRelatório de Classificação\n")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=list(generator.class_indices.keys())
        )
    )

    print("\nAcurácia:", accuracy_score(y_true, y_pred))

    print("\nMatriz de Confusão")

    print(confusion_matrix(y_true, y_pred))