# Librerias
import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork, BayesianModel
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator, PC
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score, BicScore
import math
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.io as pio
pio.renderers.default = "browser"

##
def EstimacionModelos():
    # Cargar datos y discretizar variables
    data =  pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data', header=None)
    names= ["age","sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope","ca", "thal", "num"]
    data.columns = names
    data['ca'] = pd.to_numeric(data['ca'], errors='coerce')
    data['thal'] = pd.to_numeric(data['thal'], errors='coerce')
    data = data.astype(float)

    # Se eliminan los Nan y se convierte a un arreglo de Numpy
    data = data.dropna()
    data = data.to_numpy()

    # Se estandarizan las variables para el diagnostico:
    # 0 -- No presenta heart disease
    # 1 -- mild heart disease
    # 3 -- severe heart disease
    for j in range(0, data.shape[0]):
        if data[j, 13] == 2:
            data[j, 13] = 1
        elif data[j, 13] == 4:
            data[j, 13] = 3

    # Discretizacion del colesterol
    # menos de 200 -- Deseable
    # de 200 a 239 -- En el limite superior
    # mas de 240 -- alto
    # https://www.mayoclinic.org/es-es/tests-procedures/cholesterol-test/about/pac-20384601
    for j in range(0, data.shape[0]):
        if data[j, 4] < 200:
            data[j, 4] = 0
        elif (200 <= data[j, 4] < 240):
            data[j, 4] = 1
        elif data[j, 4] >= 240:
            data[j, 4] = 2

    # Discretización de OldPeak
    # Menos de 2 - 0
    # Entre 2 y 4 - 1
    # Mayor o igual a 4 - 2
    for j in range(0, data.shape[0]):
        if data[j, 9] < 2:
            data[j, 9] = 0
        elif (2 <= data[j, 9] < 4):
            data[j, 9] = 1
        elif data[j, 9] >= 4:
            data[j, 9] = 2


    # Discretización de la edad
    # 29 a 39 -- 30
    # 40 a 49 -- 40
    # 50 a 59 -- 50
    # 60 a 69 -- 60
    # Mayor o igual a 70 -- 70
    for j in range(0, data.shape[0]):
        if (29 <= data[j, 0] < 40):
            data[j, 0] = 30
        elif (40 <= data[j, 0] < 50):
            data[j, 0] = 40
        elif (50 <= data[j, 0] < 60):
            data[j, 0] = 50
        elif (60 <= data[j, 0] < 70):
            data[j, 0] = 60
        elif data[j, 0] >= 70 :
            data[j, 0] = 70

    # Se dividen los datos entre Entrenamiento y Validación
    DataEntrenamiento = data[0:250,]

    # Se define el modelo del Proyecto 1
    # Se define la red bayesiana
    modelo_HD = BayesianNetwork([("AGE", "CHOL"), ("FBS", "CHOL"), ("CHOL", "HD"), ("THAL", "HD"), ("HD", "EXANG"),
                             ("HD", "OLDPEAK")])

    # Se definen las muestras con los datos de entrenamiento
    info = np.zeros((250,7))
    columnas = [0, 4, 5, 8, 9, 12, 13]
    nombres = ["AGE", "CHOL", "FBS", "EXANG", "OLDPEAK", "THAL", "HD"]
    for i in range(len(columnas)):
        info[:,i] = DataEntrenamiento[:,columnas[i]]
    muestras = pd.DataFrame(info, columns = nombres)

    # Estimación de las CPDs
    modelo_HD.fit(data = muestras, estimator = MaximumLikelihoodEstimator)
    # Se realiza la eliminación de variables
    ModHDP1 = VariableElimination(modelo_HD)

    # Se define el modelo por puntaje K2
    scoring_method = K2Score(data = muestras)
    esth = HillClimbSearch(data = muestras)
    estimated_modelh = esth.estimate(
        scoring_method=scoring_method, max_indegree=8, max_iter=int(1e4))

    modeloK2 = BayesianNetwork()
    edges = estimated_modelh.edges()

    modeloK2.add_edges_from(edges)
    modeloK2.fit(data = muestras, estimator = MaximumLikelihoodEstimator)
    ModK2 = VariableElimination(modeloK2)

    # Se retornan los modelos estimados
    return ModHDP1, ModK2

modelo1 = EstimacionModelos()[0]
modelo2 = EstimacionModelos()[1]

age = 60
Fbs = 1
Chol = 2
st = 1
ex = 1
tal = 7

pred = modelo1.query(["HD"], evidence = {"AGE": age, "FBS": Fbs, "CHOL": Chol, "OLDPEAK": st, "EXANG": ex, "THAL": tal})
pred2 = modelo2.query(["HD"], evidence = {"OLDPEAK": st, "EXANG": ex, "THAL": tal})


heart = ['No Heart Disease', 'Mild Heart Disease', 'Severe Heart Disease']
# Resultados predicción M1
dict2 = {'Nivel Enfermedad Cardiaca': heart, 'Probabilidad Estimada': [round(pred.values[0],2), round(pred.values[1],2), round(pred.values[2],2)]}
data = pd.DataFrame(dict2)
# Resultados predicción M2
dict22 = {'Nivel Enfermedad Cardiaca': heart, 'Probabilidad Estimada': [round(pred2.values[0],2), round(pred2.values[1],2), round(pred2.values[2],2)]}
data2 = pd.DataFrame(dict22)



# Se crea la gráfica de barras

if math.isnan(pred.values[0]) or math.isnan(pred.values[1]) or math.isnan(pred.values[2]):

    fig = make_subplots(rows=1, cols=2, subplot_titles=("No es posible calcular la probabilidad con los datos ingresados",
                                                        "Modelo Estimado por Puntake K2"))
    fig.add_trace(go.Bar(x = dict2['Nivel Enfermedad Cardiaca'], y = dict2['Probabilidad Estimada'],
                         text = dict2['Probabilidad Estimada'],
                         textposition = 'auto'), row = 1, col = 1)
    fig.update_traces(marker_color = '#EFAFAB', textfont_size = 14)

    fig.add_trace(go.Bar(x = dict22['Nivel Enfermedad Cardiaca'], y = dict22['Probabilidad Estimada'],
                         text = dict22['Probabilidad Estimada'],
                         textposition = 'auto'), row = 1, col = 2)
    fig.update_traces(marker_color = '#EFAFAB', textfont_size = 14)

    fig.update_layout(width = 1400, bargap = 0.45,
                      plot_bgcolor = "rgba(255,255,255,255)",
                      title_text = 'Probabilidad Estimada Enfermedad Cardiaca', title_x = 0.5,
                      title_font_size = 25,
                      showlegend = False)

    fig.update_xaxes(range = [-0.5, 2.5], showline = True, linewidth = 1, linecolor = 'black', mirror = True,
                     tickfont = dict(size = 15), title_text = 'Precisión : 0.76', row = 1, col = 2)
    fig.update_yaxes(showline = True, linewidth = 1, linecolor = 'black', mirror = True,
                     tickfont=dict(size = 15))


elif math.isnan(pred2.values[0]) or math.isnan(pred2.values[1]) or math.isnan(pred2.values[2]):

    fig = make_subplots(rows = 1, cols = 2,
                        subplot_titles = ("Modelo Estimado P1",
                                        "No es posible calcular la probabilidad con los datos ingresados"))
    fig.add_trace(go.Bar(x = dict2['Nivel Enfermedad Cardiaca'], y = dict2['Probabilidad Estimada'],
                         text = dict2['Probabilidad Estimada'],
                         textposition = 'auto'), row = 1, col = 1)
    fig.update_traces(marker_color = '#EFAFAB', textfont_size = 14)

    fig.add_trace(go.Bar(x = dict22['Nivel Enfermedad Cardiaca'], y = dict22['Probabilidad Estimada'],
                         text = dict22['Probabilidad Estimada'],
                         textposition = 'auto'), row = 1, col = 2)
    fig.update_traces(marker_color = '#EFAFAB', textfont_size = 14)

    fig.update_layout(width = 1400, bargap = 0.45,
                      plot_bgcolor = "rgba(255,255,255,255)",
                      title_text = 'Probabilidad Estimada Enfermedad Cardiaca', title_x = 0.5,
                      title_font_size = 25,
                      showlegend = False)

    fig.update_xaxes(range = [-0.5, 2.5], showline = True, linewidth = 1, linecolor = 'black', mirror = True,
                     tickfont = dict(size = 15), title_text = 'Precisión : 0.71', row = 1, col = 1)
    fig.update_yaxes(showline = True, linewidth = 1, linecolor = 'black', mirror = True, tickfont = dict(size = 15))


else:
    fig = make_subplots(rows = 1, cols = 2, subplot_titles = ("Modelo Estimado P1",
                                                              "Modelo Estimado por Puntake K2"))
    fig.add_trace(go.Bar(x = dict2['Nivel Enfermedad Cardiaca'], y = dict2['Probabilidad Estimada'],
                         text = dict2['Probabilidad Estimada'],
                         textposition = 'auto'), row = 1, col = 1)
    fig.update_traces(marker_color = '#EFAFAB', textfont_size = 14)

    fig.add_trace(go.Bar(x = dict22['Nivel Enfermedad Cardiaca'], y = dict22['Probabilidad Estimada'],
                         text = dict22['Probabilidad Estimada'],
                         textposition = 'auto'), row = 1, col = 2)
    fig.update_traces(marker_color = '#EFAFAB', textfont_size = 14)

    fig.update_layout(width = 1400, bargap = 0.45,
                      plot_bgcolor = "rgba(255,255,255,255)",
                      title_text = 'Probabilidad Estimada Enfermedad Cardiaca', title_x = 0.5,
                      title_font_size = 25,
                      showlegend = False)

    fig.update_xaxes(range = [-0.5, 2.5], showline = True, linewidth = 1, linecolor = 'black', mirror = True,
                     tickfont = dict(size = 15), title_text = 'Precisión : 0.71', row = 1, col = 1)

    fig.update_xaxes(range = [-0.5, 2.5], showline = True, linewidth = 1, linecolor = 'black', mirror = True,
                     tickfont = dict(size = 15), title_text = 'Precisión : 0.76', row = 1, col = 2)

    fig.update_yaxes(showline = True, linewidth = 1, linecolor = 'black', mirror = True,tickfont = dict(size = 15))

pyo.iplot(fig)