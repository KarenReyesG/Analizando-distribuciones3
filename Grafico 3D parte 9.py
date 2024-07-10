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


# Ejecutar TSNE
X_embedded = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=3).fit_transform(X)


# Crear un gráfico de dispersión 3D
fig = go.Figure()

# Agregar los puntos al gráfico
fig.add_trace(go.Scatter3d(
    x=X_embedded[y==0, 0],  # x-coordinates para 'Alive'
    y=X_embedded[y==0, 1],  # y-coordinates para 'Alive'
    z=X_embedded[y==0, 2],  # z-coordinates para 'Alive'
    mode='markers',
    marker=dict(
        size=5,
        color='green',  # color para 'Alive'
        opacity=0.5
    ),
    name='Alive'
))

fig.add_trace(go.Scatter3d(
    x=X_embedded[y==1, 0],  # x-coordinates para 'Death'
    y=X_embedded[y==1, 1],  # y-coordinates para 'Death'
    z=X_embedded[y==1, 2],  # z-coordinates para 'Death'
    mode='markers',
    marker=dict(
        size=4,
        color='coral',  # color para 'Death'
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