import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd

def plot_correlation_matrix(df):
    """
    Plotea la matriz de correlación de las columnas numéricas del DataFrame.
    """
    df_numeric = df.select_dtypes(include='number')
    corr_matrix = df_numeric.corr()

    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
    plt.title('Matriz de Correlación')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_distributions(df, num_cols_subplot=4):
    """
    Plotea histogramas y KDE para todas las columnas numéricas del DataFrame.
    """
    sns.set(style="whitegrid")
    num_columns = len(df.columns)
    num_rows_subplot = math.ceil(num_columns / num_cols_subplot)

    plt.figure(figsize=(15, num_rows_subplot * 5))

    for i, column in enumerate(df.columns, 1):
        plt.subplot(num_rows_subplot, num_cols_subplot, i)
        if df[column].dtype in ['int64', 'float64']:
            sns.histplot(df[column], kde=True, bins=30)
            plt.title(f'Distribución de {column}')
            plt.xlabel('')
        else:
            plt.title(f'No se puede graficar {column} (no numérico)')
            plt.axis('off')

    plt.tight_layout()
    plt.show()
