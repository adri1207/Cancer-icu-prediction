# Cancer ICU Prediction

## Overview
This repository contains the clinical dataset and Python code for predicting **ICU mortality** and **30-day survival** in cancer patients. The data include demographics, lab results, oncological variables, and clinical scores such as SOFA and SAPS3.

## Repository Structure
Cancer-ICU-PREDICTION/
│ main.py
│ requirements.txt
│ METADATA.yaml
│ DATASET_DESCRIPTION.md
│ README.md
│
├── data/
│ df_merged_final.xlsx # Clinical dataset
├── src/ # Python modules
│ data_exploration.py
│ preprocessing.py
│ clinical_features.py
│ visualization.py
│ descriptives_analysis.py
│ visualization_roc.py
│ modelos.py
│ shap_analysis.py
│ feature_importance.py
├── reports/ # Outputs and figures
│ comparacion_variables_train_val.csv
│ resumen_total_pacientes_por_desenlace.csv
│ resultados_modelos_ML_completo.csv
│ figures/


## Installation

### Create a Python virtual environment

python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
# source venv/bin/activate

pip install -r requirements.txt
python main.py

This will:

Load the dataset from data/df_merged_final.xlsx.

Perform missing value analysis and imputations.

Process clinical variables.

Generate visualizations: distributions, correlation matrix.

Run descriptive analyses and save CSVs in reports/.

Train predictive models and save results.

Perform SHAP analysis and generate feature importance plots.

All outputs are saved in the reports/ folder.

Citation

Please cite this preprint if you use this dataset or code:

Víctor H Nieto*1,3, Adriana C Aya2,3, Andrés F. Cardona2,3,4, Edwin Pulido2,3, Heidy Trujillo2,3, Natalia Sánchez2,3, Daniel Molano1,3, Nicolle Wagner-Gutiérrez5, Oscar Arrieta6, Christian Rolfo7, Giovanni Nigita8, Joseph Nates9.

DOI (medRxiv preprint): https://doi.org/10.64898/2026.02.02.26345349

License

This project is licensed under CC-BY 4.0.