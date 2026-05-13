"""
Pré-processamento tabular: divisão estratificada em treino, validação e teste.
"""

from sklearn.model_selection import train_test_split

from utils.config import RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE


def split_dataset(df, test_size=TEST_SIZE, validation_size=VALIDATION_SIZE):
    """
    Divide o dataset em treino, validação e teste mantendo a proporção das classes.

    Primeiro separa o conjunto de teste. Depois separa uma parte do conjunto restante
    para validação. A estratificação reduz o risco de alguma classe ficar sub-representada
    em uma das divisões.
    """

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["dx"],
        random_state=RANDOM_STATE,
    )

    validation_ratio_adjusted = validation_size / (1 - test_size)

    train_df, validation_df = train_test_split(
        train_val_df,
        test_size=validation_ratio_adjusted,
        stratify=train_val_df["dx"],
        random_state=RANDOM_STATE,
    )

    print(f"Treino: {len(train_df)} imagens")
    print(f"Validação: {len(validation_df)} imagens")
    print(f"Teste: {len(test_df)} imagens")

    return train_df, validation_df, test_df
