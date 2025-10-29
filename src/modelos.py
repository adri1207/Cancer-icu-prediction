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

# =====================================
# 4️⃣ Función para evaluar modelos
# =====================================
def evaluar_modelos(X, y, target_name, modelos):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    resultados = []
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for nombre, modelo in modelos.items():
        aucs, f1s, recalls, precisions, especificidades, accuracies = [], [], [], [], [], []
        print(f"\n🔹 Evaluando modelo: {nombre} ({target_name})")

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Imputación
            imputer = SimpleImputer(strategy='mean')
            X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
            X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

            # Escalado
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # LASSO selección de variables
            lasso = LogisticRegression(penalty='l1', solver='saga', max_iter=5000, random_state=42)
            lasso.fit(X_train_scaled, y_train)
            mask = np.abs(lasso.coef_).flatten() > 1e-5
            selected_features = X.columns[mask] if any(mask) else X.columns
            X_train_sel, X_test_sel = X_train[selected_features], X_test[selected_features]

            # --- Ajuste para XGBoost binario ---
            if nombre == 'XGBoost':
                clases_unicas = np.unique(y_train)
                print(f"Valores únicos en y_train para {nombre} ({target_name}):", clases_unicas)

                if len(clases_unicas) != 2:
                    raise ValueError(f"XGBoost solo soporta clasificación binaria, pero se detectaron {len(clases_unicas)} clases")

                # Mapear clases a 0 y 1
                clase_0, clase_1 = clases_unicas
                y_train_bin = y_train.replace({clase_0: 0, clase_1: 1})
                y_test_bin = y_test.replace({clase_0: 0, clase_1: 1})

                neg, pos = np.bincount(y_train_bin)
                modelo.set_params(scale_pos_weight=neg / pos)
            else:
                y_train_bin, y_test_bin = y_train, y_test  # Para modelos no binarios

            modelo.fit(X_train_sel, y_train_bin)
            y_pred = modelo.predict(X_test_sel)

            try:
                probas = modelo.predict_proba(X_test_sel)[:, 1]
            except:
                probas = modelo.decision_function(X_test_sel)

            # Cálculo de métricas
            auc = roc_auc_score(y_test_bin, probas) if len(np.unique(y_test_bin)) > 1 else np.nan
            f1 = f1_score(y_test_bin, y_pred)
            recall = recall_score(y_test_bin, y_pred)
            precision = precision_score(y_test_bin, y_pred)
            acc = accuracy_score(y_test_bin, y_pred)

            try:
                tn, fp, fn, tp = confusion_matrix(y_test_bin, y_pred).ravel()
                especificidad = tn / (tn + fp)
            except:
                especificidad = np.nan

            aucs.append(auc); f1s.append(f1); recalls.append(recall)
            precisions.append(precision); accuracies.append(acc); especificidades.append(especificidad)
            print(f"  Fold {fold}: {len(selected_features)} vars seleccionadas (AUROC={auc:.3f})")

        resultados.append({
            'Target': target_name, 'Model': nombre,
            'AUROC': np.nanmean(aucs), 'Accuracy': np.mean(accuracies),
            'F1': np.mean(f1s), 'Precision': np.mean(precisions),
            'Recall': np.mean(recalls), 'Specificity': np.mean(especificidades)
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
        value_vars=['AUROC', 'Accuracy', 'F1', 'Precision', 'Recall', 'Specificity']
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
