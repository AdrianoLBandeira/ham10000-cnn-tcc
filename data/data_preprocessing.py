"""
Divide dataset em treino e teste
"""

from sklearn.model_selection import train_test_split


def split_dataset(df):

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["dx"],
        random_state=42
    )

    return train_df, test_df