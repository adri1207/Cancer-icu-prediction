# visualization_roc.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# visualization_roc.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder

def graficar_curvas_roc_modelos(X, y, modelos_dict, target_name):
    """
    Genera curvas ROC promedio con 5-fold CV para múltiples modelos.
    Reentrena cada modelo desde cero para evitar errores de features.
    Guarda la figura como PNG.
    """
    # Convertir arrays a DataFrame / Series si es necesario
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"var_{i}" for i in range(X.shape[1])])
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    # Asegurar que y sea 0/1 si es binario
    if y.nunique() == 2 and sorted(y.unique()) != [0, 1]:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)

    plt.figure(figsize=(10, 8))
    mean_fpr = np.linspace(0, 1, 100)

    for nombre, modelo in modelos_dict.items():
        print(f"📈 Entrenando y generando ROC para: {nombre} ({target_name})")
        tprs = []
        aucs = []
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Pipeline con imputación + modelo
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('model', modelo.__class__(**modelo.get_params()))
            ])
            
            pipeline.fit(X_train, y_train)

            try:
                probas_ = pipeline.predict_proba(X_test)[:, 1]
            except AttributeError:
                # Si el modelo no tiene predict_proba (p.ej., SVM sin probas)
                probas_ = pipeline.decision_function(X_test)
            
            fpr, tpr, _ = roc_curve(y_test, probas_, pos_label=1)
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

        if len(aucs) == 0:
            continue

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        plt.plot(mean_fpr, mean_tpr, lw=2,
                 label=f'{nombre} (AUC = {mean_auc:.2f} ± {std_auc:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Random')
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title(f'Curves ROC - {target_name}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    filename = f"curva_roc_{target_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[✔] Gráfico guardado como: {filename}")
    plt.show()
