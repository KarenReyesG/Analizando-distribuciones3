# Visualización 3D: Análisis de Distribuciones

# Instrucciones para visualizar la distribución de tus datos utilizando t-SNE (t-Distributed Stochastic Neighbor Embedding) en un gráfico de dispersión 3D con Plotly.

# Aquí están los pasos que debes seguir:

Exportar una matriz con solo los valores de los atributos en formato de numpy array:

Utiliza df.drop(columns=['DEATH_EVENT', 'Edad_Categoria']) para eliminar las columnas que contienen información sobre si la persona murió o no y la categoría de edad.
Convierte el DataFrame a un numpy array usando df.values.
Exportar un array unidimensional con la columna objetivo DEATH_EVENT:

Extrae la columna 'DEATH_EVENT' en un array unidimensional llamado y.
Ejecutar t-SNE:

Utiliza la siguiente línea de código para calcular la reducción de dimensionalidad:

X_embedded = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=3).fit_transform(X)
Donde X_embedded será un NumPy array de dimensiones (299, 3).
Crear un gráfico de dispersión 3D con Plotly:

Crea un gráfico 3D con dos clases: 'Alive' y 'Death'.
Asigna colores diferentes a los puntos de cada clase para diferenciarlos.
Aquí tienes el código completo:


import pandas as pd
from sklearn.manifold import TSNE
import plotly.graph_objs as go

# Cargar los datos
datos = pd.read_csv('heart_failure_data_ETL.csv')

# Eliminar las columnas que no se necesitan
datos_sin_columnas = datos.drop(columns=['DEATH_EVENT', 'Edad_Categoria'])

# Convertir el DataFrame a un numpy array
X = datos_sin_columnas.values

# Extraer la columna objetivo en un array unidimensional
y = datos['DEATH_EVENT'].values

# Ejecutar t-SNE
X_embedded = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=3).fit_transform(X)

# Crear un gráfico de dispersión 3D
fig = go.Figure()

# Agregar los puntos al gráfico
fig.add_trace(go.Scatter3d(
    x=X_embedded[y==0, 0],  # Coordenadas x para 'Alive'
    y=X_embedded[y==0, 1],  # Coordenadas y para 'Alive'
    z=X_embedded[y==0, 2],  # Coordenadas z para 'Alive'
    mode='markers',
    marker=dict(
        size=5,
        color='green',  # Color para 'Alive'
        opacity=0.5
    ),
    name='Alive'
))

fig.add_trace(go.Scatter3d(
    x=X_embedded[y==1, 0],  # Coordenadas x para 'Death'
    y=X_embedded[y==1, 1],  # Coordenadas y para 'Death'
    z=X_embedded[y==1, 2],  # Coordenadas z para 'Death'
    mode='markers',
    marker=dict(
        size=4,
        color='coral',  # Color para 'Death'
        opacity=0.8
    ),
    name='Death'
))

# Configurar el layout del gráfico
fig.update_layout(
    title='TSNE: Distribución 3D de Alive vs Death',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    )
)

# Mostrar el gráfico
fig.show()
Este código generará un gráfico 3D que muestra la distribución de las clases 'Alive' y 'Death' en función de las tres dimensiones obtenidas mediante t-SNE. ¡Espero que te sea útil! Si necesitas más ayuda, no dudes en preguntar.
