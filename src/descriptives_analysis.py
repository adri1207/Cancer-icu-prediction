import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact

def comparar_para_dos_desenlaces(df, target_vars):
    """
    Compara variables numéricas y categóricas entre conjuntos de train y validación
    para cada desenlace especificado en target_vars.
    """

    resultados = []
    resumen_pacientes = []

    for target in target_vars:
        df_limpio = df[df[target].notna()]
        df_train, df_val = train_test_split(
            df_limpio, test_size=0.3, random_state=42, stratify=df_limpio[target]
        )

        resumen_pacientes.append({
            'Desenlace': target,
            'Pacientes Train': len(df_train),
            'Pacientes Validación': len(df_val),
            'Total': len(df_train) + len(df_val)
        })

        numericas = df_train.select_dtypes(include='number').columns.tolist()
        categoricas = df_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        if target in numericas: numericas.remove(target)
        if target in categoricas: categoricas.remove(target)

        # ---------------- Variables Continuas ----------------
        for var in numericas:
            try:
                med_train = df_train[var].median()
                iqr_train = df_train[var].quantile([0.25, 0.75])
                med_val = df_val[var].median()
                iqr_val = df_val[var].quantile([0.25, 0.75])
                stat, p = mannwhitneyu(df_train[var].dropna(), df_val[var].dropna())

                resumen_train = f"{med_train:.1f} [{iqr_train[0.25]:.1f}–{iqr_train[0.75]:.1f}]"
                resumen_val = f"{med_val:.1f} [{iqr_val[0.25]:.1f}–{iqr_val[0.75]:.1f}]"

                med_total = df_limpio[var].median()
                iqr_total = df_limpio[var].quantile([0.25, 0.75])
                resumen_total = f"{med_total:.1f} [{iqr_total[0.25]:.1f}–{iqr_total[0.75]:.1f}]"

                resultados.append({
                    'Desenlace': target,
                    'Variable': var,
                    'Tipo': 'Continua',
                    'Train': resumen_train,
                    'Validación': resumen_val,
                    'Total': resumen_total,
                    'p-valor': round(p, 4)
                })
            except:
                continue

        # ---------------- Variables Categóricas ----------------
        for var in categoricas:
            try:
                df_train_temp = df_train[[var]].copy()
                df_train_temp['grupo'] = 'train'
                df_val_temp = df_val[[var]].copy()
                df_val_temp['grupo'] = 'val'

                df_cat = pd.concat([df_train_temp, df_val_temp])
                tabla_cont = pd.crosstab(df_cat[var], df_cat['grupo'])

                def formatear_conteo(df):
                    total = df.sum()
                    return ", ".join([
                        f"{cat}: {df.get(cat,0)} ({(df.get(cat,0)/total*100):.1f}%)"
                        for cat in sorted(df.index)
                    ])

                resumen_train = formatear_conteo(tabla_cont['train']) if 'train' in tabla_cont else ""
                resumen_val = formatear_conteo(tabla_cont['val']) if 'val' in tabla_cont else ""
                resumen_total = formatear_conteo(tabla_cont.sum(axis=1))

                if tabla_cont.shape == (2, 2):
                    _, p = fisher_exact(tabla_cont)
                else:
                    _, p, _, _ = chi2_contingency(tabla_cont)

                resultados.append({
                    'Desenlace': target,
                    'Variable': var,
                    'Tipo': 'Categorica',
                    'Train': resumen_train,
                    'Validación': resumen_val,
                    'Total': resumen_total,
                    'p-valor': round(p, 4)
                })

            except Exception as e:
                print(f"Error procesando variable categórica {var}: {e}")
                continue

    return pd.DataFrame(resultados), pd.DataFrame(resumen_pacientes)
