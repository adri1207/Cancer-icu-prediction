# 🧠 Cancer ICU Prediction

Este proyecto tiene como objetivo desarrollar y evaluar modelos de **machine learning** para predecir:

- Mortalidad hospitalaria.
- Supervivencia a 30 días.

en pacientes críticos con cáncer ingresados a UCI.

---

## 📁 Estructura del proyecto

cancer-icu-prediction/
│
├── data/ # Archivos de datos (no subir datos sensibles)
├── notebooks/ # Notebooks de exploración y modelado
├── src/ # Código fuente del proyecto
│ ├── preprocessing.py # Limpieza e imputación de datos
│ ├── feature_engineering.py # Creación de variables y codificación
│ └── init.py
├── models/ # Modelos entrenados
├── reports/ # Resultados y figuras
│ └── figures/ # Gráficos generados
├── main.py # Script principal del pipeline
├── requirements.txt # Dependencias del proyecto
└── .gitignore


---

## ⚙️ Instalación

1. Clona este repositorio:

```bash
git clone https://github.com/adri1207/cancer-icu-prediction.git
cd cancer-icu-prediction


pip install -r requirements.txt


python main.py