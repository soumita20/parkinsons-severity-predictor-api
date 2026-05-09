import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Features to drop — not predictive
DROP_COLS = ['subject#']

# Primary and secondary targets
TARGET_MOTOR = 'motor_UPDRS'
TARGET_TOTAL = 'total_UPDRS'


def load_and_clean(filepath: str):
    """
    Load dataset, drop non-predictive columns,
    handle missing values, return cleaned dataframe.
    """
    df = pd.read_csv(filepath)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nTarget variable stats:")
    print(df[[TARGET_MOTOR, TARGET_TOTAL]].describe())

    # Drop subject ID — not a predictive feature
    df = df.drop(columns=DROP_COLS)

    return df


def split_data(df, target=TARGET_MOTOR, test_size=0.2, random_state=42):
    """
    Split into train and test sets.
    Scales features using StandardScaler — required for regression models.
    Returns X_train, X_test, y_train, y_test, scaler, feature_names
    """
    # Separate features and target
    feature_cols = [c for c in df.columns
                    if c not in [TARGET_MOTOR, TARGET_TOTAL]]
    X = df[feature_cols]
    y = df[target]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    # Scale features — critical for regression unlike TF-IDF text data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Features: {feature_cols}")

    return (X_train_scaled, X_test_scaled,
            y_train, y_test,
            scaler, feature_cols)