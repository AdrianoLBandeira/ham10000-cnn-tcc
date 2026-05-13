"""
Cálculo e salvamento das métricas de avaliação.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils.config import FIGURE_DIR, REPORT_DIR


def evaluate_model(model, generator):
    """
    Avalia o modelo no conjunto informado e salva relatório de classificação e matriz de confusão.
    """

    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    predictions = model.predict(generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = generator.classes

    class_names = list(generator.class_indices.keys())

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    accuracy = accuracy_score(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)

    print("\nRelatório de Classificação\n")
    print(report_text)
    print(f"\nAcurácia: {accuracy:.4f}")
    print("\nMatriz de Confusão")
    print(matrix)

    pd.DataFrame(report_dict).transpose().to_csv(
        os.path.join(REPORT_DIR, "classification_report.csv"),
        index=True,
    )

    with open(os.path.join(REPORT_DIR, "classification_report.txt"), "w", encoding="utf-8") as file:
        file.write(report_text)
        file.write(f"\nAcurácia: {accuracy:.4f}\n")

    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        os.path.join(REPORT_DIR, "confusion_matrix.csv"),
        index=True,
    )

    _save_confusion_matrix_plot(matrix, class_names)

    return {
        "accuracy": accuracy,
        "classification_report": report_dict,
        "confusion_matrix": matrix,
    }


def _save_confusion_matrix_plot(matrix, class_names):
    plt.figure(figsize=(8, 8))
    plt.imshow(matrix)
    plt.title("Matriz de confusão")
    plt.xlabel("Classe prevista")
    plt.ylabel("Classe real")
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(np.arange(len(class_names)), class_names)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "confusion_matrix.png"), bbox_inches="tight")
    plt.close()
