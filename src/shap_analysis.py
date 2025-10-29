# shap_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier
import shap

def entrenar_catboost_pipeline(X, y):
    """
    Entrena CatBoost con imputación de medianas.
    Devuelve el pipeline y los datos imputados.
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', CatBoostClassifier(
            auto_class_weights='Balanced',
            verbose=0,
            random_state=42
        ))
    ])
    pipeline.fit(X, y)
    
    X_imputed = pd.DataFrame(
        pipeline.named_steps['imputer'].transform(X),
        columns=X.columns,
        index=X.index
    )
    
    return pipeline, X_imputed

def calcular_shap_values(pipeline, X_imputed):
    """
    Crea explainer y calcula SHAP values para CatBoost.
    Devuelve el explainer y el array de SHAP values.
    """
    explainer = shap.TreeExplainer(pipeline.named_steps['model'])
    shap_values = explainer.shap_values(X_imputed)
    return explainer, shap_values

def plot_summary_shap(shap_values, X_imputed, title, filename):
    """
    Genera el summary plot de SHAP y lo guarda como PNG.
    """
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_imputed,
        feature_names=X_imputed.columns,
        show=False
    )
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

def plot_decision_shap(explainer, shap_values, X_imputed, title, filename, n_instances=5):
    plt.figure()
    ax = plt.gca()
    shap.decision_plot(
        base_value=explainer.expected_value,  # ✅ usar explainer en lugar de shap_values
        shap_values=shap_values[:n_instances],
        features=X_imputed.iloc[:n_instances],
        feature_names=list(X_imputed.columns),

        show=False
    )
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"[✔] SHAP decision plot guardado como: {filename}")
    plt.show()

