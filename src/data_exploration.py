import pandas as pd

""" Explorar DataSET
"""
def explorar_dataset(ruta_excel, nombre_columna_check=None):
    # Cargar archivo
    df = pd.read_excel(ruta_excel)
    
    print("="*60)
    print("✅ Primeras filas del dataset:")
    print(df.head())
    
    print("\n✅ Dimensiones (filas, columnas):")
    print(df.shape)
    
    print("\n✅ Tipos de datos por columna:")
    print(df.dtypes)
    
    print("\n✅ Lista de columnas:")
    print(list(df.columns))

    if nombre_columna_check:
        print(f"\n✅ ¿La columna '{nombre_columna_check}' está presente?:", nombre_columna_check in df.columns)

    # Revisión de valores faltantes
    print("\n✅ ¿Existen datos faltantes?:", df.isnull().values.any())

    print("\n✅ ¿Qué columnas tienen datos faltantes?:")
    print(df.isnull().any())

    print("\n✅ ¿Cuántos datos faltan por fila (muestra de filas con datos faltantes)?:")
    print(df[df.isnull().any(axis=1)])

    print("\n✅ ¿Cuántos datos faltan por columna?:")
    print(df.isnull().sum())

    print("\n✅ Total de datos faltantes en todo el dataset:", df.isnull().sum().sum())
    
    return df
