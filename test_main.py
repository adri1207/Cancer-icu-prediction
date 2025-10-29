import sys
import os

# Agrega la carpeta src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Ahora sí puedes importar tus módulos
from data_exploration import explorar_dataset
from preprocessing import analizar_faltantes, imputar_variables_continuas, limpiar_y_preparar_dataset
from clinical_features import procesar_variables_clinicas
from visualization import plot_correlation_matrix, plot_distributions
from descriptives_analysis import comparar_para_dos_desenlaces
from visualization_roc import graficar_curvas_roc_modelos
from modelos import evaluar_modelos, modelos,plot_metric_heatmap
from shap_analysis import entrenar_catboost_pipeline, calcular_shap_values, plot_summary_shap, plot_decision_shap
from feature_importance import obtener_importancia_features, plot_feature_importance
import pandas as pd
import numpy as np 
from sklearn.model_selection import StratifiedKFold 



# 1️⃣ Cargar dataset
df = explorar_dataset("df_merged_final.xlsx", nombre_columna_check="mortality")

# 2️⃣ Analizar faltantes
faltantes_ordenados = analizar_faltantes(df, umbral=20)

# 3️⃣ Imputar variables continuas
variables_a_imputar = [
    'potasioing', 'mortesper_saps3', 'creing', 'sodioing',
    'buning', 'sofa_24h', 'ecog', 'calciing', 'pco2ing', 'po2ing',
    'hco3ing', 'being', 'lactatoing', 'pafiing', 'karnofsky',
    'pting', 'pttingr', 'bilitorring', 'bding', 'biing',
    'tgoing', 'tgping',
]
df = imputar_variables_continuas(df, variables_a_imputar)

# 4️⃣ Procesar variables clínicas
df = procesar_variables_clinicas(df)

## Limpieza final

df = limpiar_y_preparar_dataset(df)

# 5️⃣ Visualización
plot_distributions(df)
plot_correlation_matrix(df)


# 6️⃣ Análisis descriptivo comparativo

print("📊 Revisión inicial de desenlaces:")
print(f"Total filas: {len(df)}")
print(f"Mortality - Total elementos: {df['mortality'].shape[0]}, NaN: {df['mortality'].isna().sum()}")
print(f"30survival - Total elementos: {df['30survival'].shape[0]}, NaN: {df['30survival'].isna().sum()}")

# Ejecutar el análisis
tabla_final, resumen_pacientes = comparar_para_dos_desenlaces(df, ['mortality', '30survival'])

# Guardar resultados
tabla_final.to_excel("comparacion_variables_train_val.xlsx", index=False)
resumen_pacientes.to_excel("resumen_total_pacientes_por_desenlace.xlsx", index=False)

print("\n✅ Resumen de pacientes por desenlace:")
print(resumen_pacientes)

print("\n🔍 Últimas variables comparadas:")
print(tabla_final.tail(10))


# 7️⃣ Preparar datasets para modelado
df_mort = df[df['mortality'].notna()].drop(columns=['30survival','record_id'])
X_mort = df_mort.drop(columns=['mortality'])
y_mort = df_mort['mortality']

df_surv = df[df['30survival'].notna()].drop(columns=['mortality','record_id'])
X_surv = df_surv.drop(columns=['30survival'])
y_surv = df_surv['30survival']

# 8️⃣ Evaluar modelos
resultados_mortalidad = evaluar_modelos(X_mort, y_mort, 'mortality', modelos)
resultados_sobrevida = evaluar_modelos(X_surv, y_surv, '30-day survival', modelos)

# 9️⃣  Guardar resultados
resultados_finales = pd.concat([resultados_mortalidad, resultados_sobrevida], ignore_index=True)
resultados_finales.to_excel("resultados_modelos_ML_completo.xlsx", index=False)
print(resultados_finales)

#  Heatmap de métricas
plot_metric_heatmap(resultados_finales)


# 1️⃣0️⃣  Curvas ROC
# Asegurar que las etiquetas sean 0 y 1
y_mort = y_mort.replace({2: 1, 1: 0}) if y_mort.max() == 2 else y_mort
y_surv = y_surv.replace({2: 1, 1: 0}) if y_surv.max() == 2 else y_surv


graficar_curvas_roc_modelos(X_mort, y_mort, modelos, "Mortality")
graficar_curvas_roc_modelos(X_surv, y_surv, modelos, "30-day survival")

## 1️⃣1️⃣ SHAP Analysis for Mortality Prediction with CatBoost  ##

from src.shap_analysis import (
    entrenar_catboost_pipeline,
    calcular_shap_values,
    plot_summary_shap,
    plot_decision_shap
)

# --- Mortalidad ---
cat_pipeline_mort, X_mort_imputed = entrenar_catboost_pipeline(X_mort, y_mort)
explainer_mort, shap_values_mort = calcular_shap_values(cat_pipeline_mort, X_mort_imputed)

# SHAP summary plot
plot_summary_shap(
    shap_values_mort,
    X_mort_imputed,
    "Figure 4A: SHAP Summary Plot - Mortality Model",
    "shap_summary_mortality.png"
)

# SHAP decision plot
plot_decision_shap(
    explainer_mort,
    shap_values_mort,
    X_mort_imputed,
    "Figure 4C: SHAP Decision Plot - Mortality Model",
    "shap_decision_mortality.png"
)

# --- Sobrevida 30 días ---
cat_pipeline_surv, X_surv_imputed = entrenar_catboost_pipeline(X_surv, y_surv)
explainer_surv, shap_values_surv = calcular_shap_values(cat_pipeline_surv, X_surv_imputed)

# SHAP summary plot
plot_summary_shap(
    shap_values_surv,
    X_surv_imputed,
    "Figure 4B: SHAP Summary Plot - 30 Day Survival Model",
    "shap_summary_30day_survival.png"
)

# SHAP decision plot
plot_decision_shap(
    explainer_surv,
    shap_values_surv,
    X_surv_imputed,
    "Figure 4D: SHAP Decision Plot - 30-Day Survival Model",
    "shap_decision_30day_survival.png"
)


# 1️⃣2️⃣ Feature importance Mortalidad
df_importancia_mort = obtener_importancia_features(cat_pipeline_mort, X_mort)
plot_feature_importance(
    df_importancia_mort,
    title='Figure 3A: Feature Importance - CatBoost Mortality Model',
    filename='feature_importance_mortality_academic.png'
)

# Feature importance Sobrevida 30 días
df_importancia_surv = obtener_importancia_features(cat_pipeline_surv, X_surv)
plot_feature_importance(
    df_importancia_surv,
    title='Figure 3B: Feature Importance - CatBoost 30-Day Survival Model',
    filename='feature_importance_30day_survival_academic.png'
)
