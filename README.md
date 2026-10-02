# Telecom - Predicción de cancelación de clientes

Proyecto de Machine Learning para predecir qué clientes de una empresa de telecomunicaciones van a cancelar su contrato (churn), para que la empresa pueda ofrecerles promociones antes de que se vayan.

## Resultados

El mejor modelo fue **XGBoost** (ajustado con GridSearch y con un umbral de 0.4). Estos son sus resultados con los datos de prueba:

| Métrica | Valor |
|---|---|
| AUC-ROC | 0.91 |
| F1 | 0.73 |
| Recall | 0.73 |
| Precisión | 0.74 |

El modelo encuentra a 3 de cada 4 clientes que cancelan. Los clientes con más riesgo son los que tienen contrato mes a mes, fibra óptica y poco tiempo con la empresa.

## Datos

Cuatro archivos en `data/raw/` que se unen por `customerID` (7043 clientes):

- `contract.csv`: fechas del contrato, tipo de contrato, método de pago y cargos.
- `personal.csv`: género, si es adulto mayor, si tiene pareja y dependientes.
- `internet.csv`: tipo de internet y servicios adicionales (seguridad, backup, soporte, streaming).
- `phone.csv`: si el cliente tiene varias líneas.

## Qué hice

1. Limpié y uní las tablas, y creé variables nuevas como los meses que lleva el cliente y la cantidad de servicios que tiene.
2. Hice un análisis exploratorio para ver qué tipo de clientes cancela más.
3. Comparé regresión logística, Random Forest y XGBoost con validación cruzada (con y sin SMOTE) contra un modelo base.
4. Ajusté XGBoost con GridSearch y elegí el umbral de decisión.
5. Evalué el modelo final una sola vez con los datos de prueba.

**Algo que corregí:** en la primera versión usaba una fecha de corte equivocada (2021-02-01 en lugar de 2020-02-01) para calcular la antigüedad de los clientes. Eso hacía que el modelo pareciera mucho mejor de lo que era (F1 de 0.87). Al corregirlo, las métricas bajaron pero ahora son reales.

## Cómo ejecutarlo

```bash
git clone https://github.com/dixonpa/telecom.git
cd telecom
python -m venv .venv
.venv\Scripts\activate        # en Windows
source .venv/bin/activate     # en Mac/Linux
pip install -r requirements.txt
jupyter notebook notebooks/telecom_churn.ipynb
```

## Herramientas

Python, pandas, scikit-learn, XGBoost, imbalanced-learn, matplotlib, seaborn.
