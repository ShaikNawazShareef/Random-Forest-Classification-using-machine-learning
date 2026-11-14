# -*- coding: utf-8 -*-
"""Train and evaluate a random forest classifier for the car evaluation dataset."""

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")


def load_data():
    data_path = Path(__file__).resolve().parent / "car_evaluation.csv"
    column_names = [
        "buying",
        "maint",
        "doors",
        "persons",
        "lug_boot",
        "safety",
        "class",
    ]
    df = pd.read_csv(data_path, header=None, names=column_names)
    print("Dataset shape:", df.shape)
    print("\nFirst rows:\n", df.head())
    print("\nMissing values:\n", df.isnull().sum())
    return df


def encode_features(X_train, X_test):
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train_encoded = pd.DataFrame(
        encoder.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_encoded = pd.DataFrame(
        encoder.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_encoded, X_test_encoded


def evaluate_model(X_train, X_test, y_train, y_test, label):
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy score {label}: {accuracy:0.4f}")
    return model, predictions


def main():
    df = load_data()

    for col in df.columns:
        print(f"\nValue counts for {col}:\n{df[col].value_counts()}")

    X = df.drop(columns=["class"])
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    X_train_encoded, X_test_encoded = encode_features(X_train, X_test)

    model, predictions = evaluate_model(
        X_train_encoded,
        X_test_encoded,
        y_train,
        y_test,
        "with all features",
    )

    feature_scores = pd.Series(
        model.feature_importances_, index=X_train_encoded.columns
    ).sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=feature_scores.values, y=feature_scores.index)
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title("Visualizing Important Features")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

    X_no_doors = df.drop(columns=["class", "doors"])
    y_no_doors = df["class"]

    X_train_no_doors, X_test_no_doors, y_train_no_doors, y_test_no_doors = train_test_split(
        X_no_doors, y_no_doors, test_size=0.33, random_state=42
    )

    X_train_no_doors_enc, X_test_no_doors_enc = encode_features(
        X_train_no_doors, X_test_no_doors
    )

    _, y_pred_no_doors = evaluate_model(
        X_train_no_doors_enc,
        X_test_no_doors_enc,
        y_train_no_doors,
        y_test_no_doors,
        "with doors variable removed",
    )

    cm = confusion_matrix(y_test_no_doors, y_pred_no_doors)
    print("\nConfusion matrix:\n", cm)
    print("\nClassification report:\n")
    print(classification_report(y_test_no_doors, y_pred_no_doors))


if __name__ == "__main__":
    main()
