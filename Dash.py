## Alejandra Erazo / Juliana Mendoza
# Dash
import math

##
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
import math

# Se crea una función que estima el modelo completo con las muestras
def ModeloCalculado():
    # Se leen los datos
    data = pd.read_csv("Proyecto 1/processed.cleveland.data", sep=",")
    data_names = open("Proyecto 1/heart-disease.names").read()
    names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca",
             "thal", "num"]
    data.columns = names
    data['ca'] = pd.to_numeric(data['ca'], errors='coerce')
    data['thal'] = pd.to_numeric(data['thal'], errors='coerce')
    data = data.astype(float)

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
        elif data[j, 0] >= 70:
            data[j, 0] = 70

    # Se define el modelo

    # Se define la red bayesiana
    modelo_HD = BayesianNetwork([("AGE", "CHOL"), ("FBS", "CHOL"), ("CHOL", "HD"), ("THAL", "HD"), ("HD", "EXANG"),
                                 ("HD", "OLDPEAK")])

    # Se definen las muestras
    info = np.zeros((296, 7))
    columnas = [0, 4, 5, 8, 9, 12, 13]
    nombres = ["AGE", "CHOL", "FBS", "EXANG", "OLDPEAK", "THAL", "HD"]
    for i in range(len(columnas)):
        info[:, i] = data[:, columnas[i]]
    muestras = pd.DataFrame(info, columns=nombres)

    # Estimación de las CPDs
    modelo_HD.fit(data=muestras, estimator=MaximumLikelihoodEstimator)
    modelo_HD.check_model()

    # Eliminación de variables
    infer = VariableElimination(modelo_HD)
    return infer

modelo_prediccion = ModeloCalculado()

### CREACIÓN DEL DASH ----------------------------------------------------------------------------------------------------

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Se definen los colores
colors = {'background': '#E6CDFA','color': '#521383'}

# Se crea el layout del Dash
app.layout = html.Div(children=[

    # Título
    html.H1('Sistema de Predicción de Enfermedad Cardíaca', style={'backgroundColor': colors['background'],
                                                                   'textAlign': 'center'}),

    # Subtítulo con exlpicación del sistema
    html.Div(html.H6('Este sistema permite realizar predicciones del riesgo de sufrir una enfermedad cardíaca para determinar '
             'el proceso adecuado a seguir, en busca del bienestar del paciente. Para esto, se tienen en cuenta los '
             'siguientes parámetros:')),

    # Explicación de los parámetros utilizads
    html.Div(html.H6(dcc.Markdown('''
    * **Edad (Age):** Edad del paciente (años).
    * **Glucosa (Fbs):** Nivel de glucosa en sangre en ayunas mayor a 120 mg/dL.
    * **Colesterol (Chol):** Valor de colesterol total en sangre (mg/dL).
    * **ST (Oldpeak):** Depresión del ST inducida por el ejercicio en relación con el reposo.
    * **Angina (Exang):** Angina inducida por el ejercicio. 
    * **Talasemia (Thal):** Tipo de talasemia.
    '''))),

    # Sección que indica la instrucción a seguir
    html.Div(html.H4('Seleccione los valores de los parámetros'), style={'backgroundColor': colors['background'],
                                                                   'textAlign': 'center'}),


    # Se definen los parametros con los valores que pueden tomar:
    html.Div([
    html.Div(html.H6("Edad (Age)", style={"color": "#521383"})),
    html.Div('''30: 29 a 39 años / 40: 40 a 49 años / 50: 50 a 59 años / 60: 60 a 69 años / 70: Mayor de 70 años'''),
    html.Div([
        dcc.Dropdown(
            id='Edad',
            options=[{'label': i, 'value': i} for i in [30, 40, 50, 60, 70]])], style={'width': '35%','display': 'inline-block'}),


    html.Div(html.H6("Glucosa (Fbs)", style={"color": "#521383"})),
    html.Div("0: No / 1: Sí"),
    html.Div([
        dcc.Dropdown(
            id='Glucosa',
            options=[{'label': i, 'value': i} for i in [0, 1]])], style={'width': '35%','display': 'inline-block'})], style= {'columnCount': 2}),

    html.Div([
    html.Div(html.H6("Colesterol (Chol)", style={"color": "#521383"})),
    html.Div("0: menos de 200 / 1: Entre 200 y 239 / 2: Mayor o igual a 240"),
    html.Div([
        dcc.Dropdown(
            id='Colesterol',
            options=[{'label': i, 'value': i} for i in [0, 1, 2]])], style={'width': '35%','display': 'inline-block'}),


    html.Div(html.H6("ST (Oldpeak)", style={"color": "#521383"})),
    html.Div("0: Menos de 2 / 1: Entre 2 y 4 / 2: Mayor o igual a 4"),
    html.Div([
        dcc.Dropdown(
            id='ST',
            options=[{'label': i, 'value': i} for i in [0, 1, 2]])], style={'width': '35%','display': 'inline-block'})], style= {'columnCount': 2}),

    html.Div([
    html.Div(html.H6("Angina (Exang)", style={"color": "#521383"})),
    html.Div("0: No / 1: Sí"),
    html.Div([
        dcc.Dropdown(
            id='Ex',
            options=[{'label': i, 'value': i} for i in [0, 1]])], style={'width': '35%','display': 'inline-block'}),

    html.Div(html.H6("Talasemia (Thal)", style={"color": "#521383"})),
    html.Div("3: Normal / 6: Defecto fijo / 7: Defecto reversible"),
    html.Div([
        dcc.Dropdown(
            id='Talasemia',
            options=[{'label': i, 'value': i} for i in [3, 6, 7]])], style={'width': '35%','display': 'inline-block'})], style= {'columnCount': 2}),

    # Se crea el botón
    html.Div([
        html.Br(),
        html.Br(),
        html.Button('Realizar predicción', id='button', n_clicks=0),
        dcc.Interval(id='interval', interval=500)]),

    # Se crea la gráfica
    html.Div([
    html.Br(),
    html.Br(),
    dcc.Graph(id='graph-prob')]),

])

# Función de Callback
@app.callback(
    Output('graph-prob', "figure"),
    [Input('button', "n_clicks")],
    [State('Edad', 'value'),
     State('Glucosa', 'value'),
     State('Colesterol', 'value'),
     State('ST', 'value'),
     State('Ex', 'value'),
     State('Talasemia', 'value')],prevent_initial_call=True)

# Función para crear y actualizar la gráfica
def update_figure(n_clicks, age, Fbs, Chol, st, ex, tal):
    # Se define el modelo
    modelo = ModeloCalculado()
    # Se realiza la predicción a partir de los parámetros obtenidos
    pred = modelo.query(["HD"], evidence={"AGE": age, "FBS": Fbs, "CHOL": Chol, "OLDPEAK": st, "EXANG": ex, "THAL": tal})
    valores1 = round(pred.values[0],2)
    valores2 = round(pred.values[1],2)
    valores3 = round(pred.values[2],2)
    heart = ['No Heart Disease', 'Mild Heart Disease', 'Severe Heart Disease']
    dict2 = {'Nivel Enfermedad Cardiaca': heart, 'Probabilidad Estimada': [valores1, valores2, valores3] }
    data = pd.DataFrame(dict2)

    # Se crea la gráfica de barras
    fig = px.bar(data, x='Nivel Enfermedad Cardiaca', y='Probabilidad Estimada', height=500, text_auto=True)
    fig.update_traces(marker_color='thistle')
    if math.isnan(valores1) or math.isnan(valores2) or math.isnan(valores3):
        fig.update_layout(width=900, bargap=0.6,
                          plot_bgcolor="rgba(255,255,255,255)",
                          title_text='No es posible calcular la probabilidad con los datos ingresados', title_x=0.5)
    else:
        fig.update_layout(width = 900, bargap = 0.6,
                      plot_bgcolor="rgba(255,255,255,255)",
                      title_text='Probabilidad Estimada Enfermedad Cardiaca', title_x=0.5)

    fig.update_xaxes(range=[-0.5, 2.5], showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False, port=9877)