import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score,
    accuracy_score, confusion_matrix
)
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibrationDisplay

# =====================================
# 4️⃣ Función para evaluar modelos
# =====================================


def calcular_calibracion_detallada(y_true, y_probs):
    """Función auxiliar para obtener slope e intercept"""
    # Evitar log(0) o log(1)
    y_probs = np.clip(y_probs, 1e-10, 1 - 1e-10)
    logit_probs = np.log(y_probs / (1 - y_probs)).reshape(-1, 1)
    
    lr_calib = LogisticRegression()
    lr_calib.fit(logit_probs, y_true)
    
    slope = lr_calib.coef_[0][0]
    intercept = lr_calib.intercept_[0]
    return slope, intercept

def evaluar_modelos(X, y, target_name, modelos):
    if isinstance(X, np.ndarray): X = pd.DataFrame(X)
    if isinstance(y, np.ndarray): y = pd.Series(y)

    resultados = []
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for nombre, modelo in modelos.items():
        aucs, f1s, recalls, precisions, especificidades, accuracies, briers = [], [], [], [], [], [], []
        slopes, intercepts = [], [] # <-- Listas nuevas
        
        print(f"\n🔹 Evaluando modelo: {nombre} ({target_name})")

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # ... [Tu código de Imputación, Escalado y LASSO igual] ...
            imputer = SimpleImputer(strategy='mean')
            X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
            X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            lasso = LogisticRegression(penalty='l1', solver='saga', max_iter=5000, random_state=42)
            lasso.fit(X_train_scaled, y_train)
            mask = np.abs(lasso.coef_).flatten() > 1e-5
            selected_features = X.columns[mask] if any(mask) else X.columns
            X_train_sel, X_test_sel = X_train[selected_features], X_test[selected_features]

            # --- Manejo de clases binarias ---
            y_train_bin, y_test_bin = y_train, y_test
            if nombre == 'XGBoost':
                clases_unicas = np.unique(y_train)
                y_train_bin = y_train.replace({clases_unicas[0]: 0, clases_unicas[1]: 1})
                y_test_bin = y_test.replace({clases_unicas[0]: 0, clases_unicas[1]: 1})
                neg, pos = np.bincount(y_train_bin)
                modelo.set_params(scale_pos_weight=neg / pos)

            # Entrenamiento y Predicción
            modelo.fit(X_train_sel, y_train_bin)
            y_pred = modelo.predict(X_test_sel)

            try:
                probas = modelo.predict_proba(X_test_sel)[:, 1]
            except:
                probas = modelo.decision_function(X_test_sel)

            # --- MÉTRICAS DE CALIBRACIÓN ---
            brier = brier_score_loss(y_test_bin, probas)
            briers.append(brier)
            
            # Cálculo de Slope e Intercept por fold
            try:
                s, i = calcular_calibracion_detallada(y_test_bin, probas)
                slopes.append(s)
                intercepts.append(i)
            except:
                slopes.append(np.nan)
                intercepts.append(np.nan)

            # --- Resto de métricas ---
            aucs.append(roc_auc_score(y_test_bin, probas))
            f1s.append(f1_score(y_test_bin, y_pred))
            accuracies.append(accuracy_score(y_test_bin, y_pred))
            tn, fp, fn, tp = confusion_matrix(y_test_bin, y_pred).ravel()
            especificidades.append(tn / (tn + fp))
            precisions.append(precision_score(y_test_bin, y_pred))
            recalls.append(recall_score(y_test_bin, y_pred))

        resultados.append({
            'Target': target_name, 
            'Model': nombre,
            'AUROC': np.nanmean(aucs), 
            'Accuracy': np.mean(accuracies),
            'F1': np.mean(f1s), 
            'Precision': np.mean(precisions),
            'Recall': np.mean(recalls), 
            'Specificity': np.mean(especificidades),
            'Brier Score': np.mean(briers),
            'Calib Slope': np.nanmean(slopes),     # <-- Nueva columna
            'Calib Intercept': np.nanmean(intercepts) # <-- Nueva columna
        })

    return pd.DataFrame(resultados)
# =====================================
# 5️⃣ Definir modelos
# =====================================
modelos = {
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'Extra Trees': ExtraTreesClassifier(class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'LightGBM': LGBMClassifier(class_weight='balanced', random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(scale_pos_weight=1, use_label_encoder=False, eval_metric='logloss', random_state=42),
    'CatBoost': CatBoostClassifier(auto_class_weights='Balanced', verbose=0, random_state=42),
}

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_metric_heatmap(resultados_finales):
    """
    Genera un heatmap comparando métricas por modelo y desenlace.
    """
    df_melt = resultados_finales.melt(
        id_vars=['Target', 'Model'],
        value_vars=['AUROC', 'Accuracy', 'F1', 'Precision', 'Recall', 'Specificity', 'Brier Score', 'Calib Slope', 'Calib Intercept'],
    )

    heatmap_data = df_melt.pivot_table(
        index='Model',
        columns=['Target', 'variable'],
        values='value'
    )

    plt.figure(figsize=(16, 10))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Metric value'})
    plt.title("Figure 2A. Comparison of metrics by model and target", fontsize=16)
    plt.tight_layout()
    plt.savefig('heatmap_metrics_models_targets.png', dpi=300)
    plt.show()

# Grafica de calibration curves

def plot_calibration_curves(modelos_entrenados, X_test_dict, y_test_dict, target_name):
    """
    Genera curvas de calibración para comparar los modelos principales.
    """
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    
    # Dibujar la línea perfecta
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for nombre in ['CatBoost', 'Logistic Regression', 'XGBoost']: # Comparamos los más importantes
        if nombre in modelos_entrenados:
            modelo = modelos_entrenados[nombre]
            # Usar los datos de test que correspondan a ese target
            X_test = X_test_dict[target_name]
            y_test = y_test_dict[target_name]
            
            # Asegurarse de usar las mismas variables que el modelo final
            CalibrationDisplay.from_estimator(
                modelo, X_test, y_test, 
                name=nombre, ax=ax, n_bins=10
            )

    ax.set_title(f"Figure 2C. Calibration Plots for {target_name}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'calibration_{target_name}.png', dpi=300)
    plt.show()