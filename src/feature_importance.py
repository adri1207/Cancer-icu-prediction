# feature_importance.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def obtener_importancia_features(pipeline, X):
    """
    Extrae la importancia de features de un modelo CatBoost dentro de un pipeline.
    Devuelve un DataFrame ordenado por importancia descendente.
    """
    importancia = pipeline.named_steps['model'].get_feature_importance()
    features = X.columns

    df_importancia = pd.DataFrame({
        'feature': features,
        'importance': importancia
    }).sort_values(by='importance', ascending=False)
    
    return df_importancia

def plot_feature_importance(df_importancia, title, filename):
    """
    Grafica feature importance con degradé azul académico y valores al final de las barras.
    """
    plt.figure(figsize=(10, 8))
    
    norm = df_importancia['importance'] / df_importancia['importance'].max()
    colors = cm.Blues(norm)  # degradé azul
    
    bars = plt.barh(df_importancia['feature'], df_importancia['importance'], color=colors, edgecolor='none')
    plt.gca().invert_yaxis()
    plt.title(title, fontsize=14)
    plt.xlabel('Feature Importance')
    plt.grid(False)
    
    # Etiquetas con valores
    for bar in bars:
        width = bar.get_width()
        plt.text(width + df_importancia['importance'].max() * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f'{width:.2f}',
                 va='center',
                 fontsize=9)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
