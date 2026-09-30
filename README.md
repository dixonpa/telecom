# Telecom: predicción de cancelación de clientes (churn)

Modelo de machine learning que predice qué clientes de un operador de telecomunicaciones van a cancelar su contrato, para que la empresa pueda ofrecerles incentivos de retención antes de que se vayan.

## 📊 Resultados

| Modelo (conjunto de prueba) | AUC-ROC | F1 | Recall | Precisión | Exactitud |
|---|---|---|---|---|---|
| Línea base (`DummyClassifier`) | 0.49 | 0.25 | 0.25 | 0.24 | 0.60 |
| **XGBoost (umbral 0.40)** | **0.90** | **0.73** | **0.73** | **0.73** | **0.86** |

- El modelo detecta **3 de cada 4 clientes que cancelan**, con una precisión del 73 %.
- Principales factores de riesgo: **contrato mes a mes**, **fibra óptica** y **poca antigüedad** como cliente.

## 🗃️ Datos

Cuatro archivos CSV en `data/raw/`, unidos por `customerID` (7 043 clientes; datos extraídos el 2020-02-01):

| Archivo | Contenido |
|---|---|
| `contract.csv` | Fechas de inicio y fin, tipo de contrato, facturación electrónica, método de pago y cargos |
| `personal.csv` | Género, adulto mayor, pareja y dependientes |
| `internet.csv` | Tipo de conexión y servicios adicionales (seguridad, backup, soporte técnico, streaming…) |
| `phone.csv` | Si el cliente tiene varias líneas telefónicas |

La variable objetivo `churn` vale 1 si el cliente tiene fecha de fin de contrato (26.5 % de los clientes).

## ⚙️ Metodología

1. **Limpieza y unión** de las cuatro tablas; tratamiento de `TotalCharges` vacíos y de servicios no contratados.
2. **Ingeniería de características:** antigüedad en meses (calculada con la fecha de corte de los datos), número de servicios adicionales e indicadores de internet y teléfono.
3. **Análisis exploratorio** de la tasa de cancelación por tipo de contrato, servicio, método de pago y cargos.
4. **Modelado con `Pipeline`** (escalado + one-hot encoding + modelo), para evitar fugas de datos.
5. **Comparación con validación cruzada estratificada (5 pliegues)** de regresión logística, Random Forest y XGBoost, con y sin SMOTE, frente a una línea base.
6. **Ajuste de hiperparámetros** (`RandomizedSearchCV`) y **del umbral de decisión** para maximizar el F1.
7. **Evaluación final única** en un conjunto de prueba reservado (20 %).

## 📁 Estructura del proyecto

```
telecom/
├── data/
│   └── raw/                 # Datos originales (CSV)
├── notebooks/
│   └── telecom_churn.ipynb  # Análisis completo y modelado
├── requirements.txt         # Dependencias con versiones fijas
└── README.md
```

## 🚀 Cómo ejecutarlo

Requisitos: Python 3.11 o superior.

```bash
git clone https://github.com/dixonpa/telecom.git
cd telecom
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/telecom_churn.ipynb
```

## 🛠️ Tecnologías

Python · pandas · NumPy · scikit-learn · imbalanced-learn · XGBoost · Matplotlib · seaborn · Jupyter

## 🔭 Próximos pasos

- Validar el modelo con datos de periodos posteriores (todas las cancelaciones del dataset ocurren entre octubre de 2019 y enero de 2020).
- Calibrar las probabilidades y probar LightGBM o CatBoost.
- Elegir el umbral según el beneficio económico esperado de las campañas de retención.
