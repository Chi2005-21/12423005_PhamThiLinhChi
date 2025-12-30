import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def load_data(path="data/credit_risk_dataset.csv"):
    return pd.read_csv(path)


def basic_info(df):
    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isna().sum())
    print("\nInfo:")
    df.info()
    print("\nDescribe:")
    print(df.describe())


def plot_target(df):
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x="loan_status")
    plt.title("Phân phối biến mục tiêu loan_status")
    plt.show()


def plot_categorical(df, categorical_cols):
    for col in categorical_cols:
        plt.figure(figsize=(6,4))
        sns.countplot(
            data=df,
            x=col,
            order=df[col].value_counts().index,
            color="pink"
        )
        plt.xticks(rotation=30)
        plt.title(f"Phân phối {col}")
        plt.show()


def plot_default_rate(df, categorical_cols):
    for col in categorical_cols:
        plt.figure(figsize=(6,4))
        sns.barplot(
            data=df,
            x=col,
            y="loan_status",
            order=df[col].value_counts().index,
            errorbar=None,
            color="steelblue"
        )
        plt.xticks(rotation=30)
        plt.ylabel("Tỷ lệ vỡ nợ")
        plt.title(f"Tỷ lệ vỡ nợ theo {col}")
        plt.show()


def plot_numeric(df, numeric_cols):
    for col in numeric_cols:
        fig, ax = plt.subplots(1, 2, figsize=(12,4))
        sns.histplot(df[col], kde=True, ax=ax[0])
        ax[0].set_title(f"Phân phối {col}")
        sns.boxplot(x=df[col], ax=ax[1])
        ax[1].set_title(f"Boxplot {col}")
        plt.show()
