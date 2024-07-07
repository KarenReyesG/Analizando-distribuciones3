import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import TSNE
import plotly.express as px

# Cargar el archivo CSV
file_path = 'heart_failure_data_ETL.csv'
data = pd.read_csv(file_path)

# Definir las variables
variables_predictivas = ['age', 'anaemia', 'diabetes', 'high_blood_pressure', 'smoking', 'serum_creatinine']
variable_objetivo = 'DEATH_EVENT'

# Preparar los datos para TSNE
X = data[variables_predictivas].values
y = data[variable_objetivo].values

# Reducción de dimensionalidad con TSNE
tsne = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=30)
X_embedded = tsne.fit_transform(X)

# Visualización 3D con Plotly
fig = px.scatter_3d(
    X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2],
    color=y,
    symbol='o',
    size=10,
    title='Visualización 3D con TSNE',
    labels={'x': 'Componente 1', 'y': 'Componente 2', 'z': 'Componente 3'}
)
fig.show()