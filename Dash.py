import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
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
import plotly.io as pio
pio.renderers.default = "browser"
import psycopg2
from sqlalchemy import create_engine, text

engine=create_engine('postgresql://postgres:proyecto2@datap2.colrzll4geas.us-east-1.rds.amazonaws.com:5432/datap2')
data0 = pd.read_sql(text("SELECT * FROM datap2"), con=engine.connect())
data0 = data0.to_numpy()

# Se estandarizan las variables para el diagnostico:
    # 0 -- No presenta heart disease
    # 1 -- mild heart disease
    # 3 -- severe heart disease
for j in range(0, data0.shape[0]):
    if data0[j, 13] == 2:
        data0[j, 13] = 1
    elif data0[j, 13] == 4:
        data0[j, 13] = 3

info = np.zeros((297, 7))
columnas = [0, 4, 5, 8, 9, 12, 13]
nombres = ["AGE", "CHOL", "FBS", "EXANG", "OLDPEAK", "THAL", "HD"]
for i in range(len(columnas)):
    info[:, i] = data0[:, columnas[i]]
data1 = pd.DataFrame(info, columns=nombres)


def EstimacionModelos():
    engine = create_engine('postgresql://postgres:proyecto2@datap2.colrzll4geas.us-east-1.rds.amazonaws.com:5432/datap2')
    data = pd.read_sql(text("SELECT * FROM datap2"), con=engine.connect())
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
    info = np.zeros((297, 7))
    columnas = [0, 4, 5, 8, 9, 12, 13]
    nombres = ["AGE", "CHOL", "FBS", "EXANG", "OLDPEAK", "THAL", "HD"]
    for i in range(len(columnas)):
        info[:, i] = data[:, columnas[i]]
    muestras = pd.DataFrame(info[0:250,], columns=nombres)


    # Estimación de las CPDs
    modelo_HD.fit(data=muestras, estimator=MaximumLikelihoodEstimator)
    # Eliminación de variables
    ModHDP1 = VariableElimination(modelo_HD)


    # Se define el modelo por puntaje K2
    scoring_method = K2Score(data = muestras)
    esth = HillClimbSearch(data = muestras)
    estimated_modelh = esth.estimate(scoring_method=scoring_method, max_indegree=8, max_iter=int(1e4))

    modeloK2 = BayesianNetwork()
    edges = estimated_modelh.edges()

    modeloK2.add_edges_from(edges)
    modeloK2.fit(data = muestras, estimator = MaximumLikelihoodEstimator)
    ModK2 = VariableElimination(modeloK2)

    # Se retornan los modelos estimados
    return ModHDP1, ModK2

# VISUALIZACIONES

# Visualizacion 1
fig_v1 = px.scatter(x=data1['AGE'], y=data1['CHOL'], trendline="ols", trendline_color_override= 'palevioletred', labels={'x':'Edad (Años)', 'y':'Colesterol (mg/dL)'})
fig_v1.update_traces(marker_color='lightpink')
fig_v1.update_layout(width=1000,plot_bgcolor="rgba(255,255,255,255)",title_text='Colesterol en función de la Edad', title_x=0.5, title_font_size=20)
fig_v1.update_xaxes( showline=True, linewidth=1, linecolor='black', mirror=True)
fig_v1.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

# Visualizacion 2
df = data1[["FBS", "EXANG", "HD"]]
df = df.to_numpy()
fbs0, fbs1, fbs3 = 0, 0, 0
exang0, exang1, exang3  = 0, 0, 0
for i in range(len(df)):
    if df[i,2] == 0:
        if df[i,0] == 1:
            fbs0 += 1
        if df[i,1] == 1:
            exang0 += 1
    if df[i,2] == 1:
        if df[i,0] == 1:
            fbs1 += 1
        if df[i,1] == 1:
            exang1 += 1
    if df[i,2] == 3:
        if df[i,0] == 1:
            fbs3 += 1
        if df[i,1] == 1:
            exang3 += 1

hd = ['No Enfermedad Cardiaca','Enfermedad Cardiaca Leve','Enfermedad Cardiaca Severa']
fbs = [fbs0, fbs1, fbs3]
exang = [exang0, exang1, exang3]
fig_v2 = go.Figure(data=[go.Bar(name='FBS = 1', x=hd, y=fbs, marker=dict(color='palevioletred')),go.Bar(name='EXANG = 1', x=hd, y=exang, marker=dict(color='lightpink'))])
fig_v2.update_layout(width=1000,barmode='group', yaxis=dict(title='Frecuencia'),  plot_bgcolor="rgba(255,255,255,255)"
                  , title='Frecuencia de Glucosa Alta y Angina Inducida por el Ejercicio', title_x=0.5, title_font_size=20,
                  legend_font_size = 16, xaxis = {'tickfont': {'size': 15}})
fig_v2.update_xaxes(range=[-0.5, 2.5], showline=True, linewidth=1, linecolor='black', mirror=True)
fig_v2.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

# Visualizacion 3
df2 = data1[["THAL","HD"]]
df2 = df2.to_numpy()

HD03, HD06, HD07, HD13, HD16, HD17, HD33, HD36, HD37 = 0,0, 0,0, 0,0, 0,0, 0

for i in range(len(df2)):
    if df2[i,0] == 3:
        if df2[i,1] == 0:
            HD03 += 1
        if df2[i,1] == 1:
            HD13 += 1
        if df2[i,1] == 3:
            HD33 += 1
    if df2[i,0] == 6:
        if df2[i,1] == 0:
            HD06 += 1
        if df2[i,1] == 1:
            HD16 += 1
        if df2[i,1] == 3:
            HD36 += 1
    if df2[i,0] == 7:
        if df2[i,1] == 0:
            HD07 += 1
        if df2[i,1] == 1:
            HD17 += 1
        if df2[i,1] == 3:
            HD37 += 1

Tal = ['Normal', 'Defecto Fijo', 'Defecto Reversible']
HD0 = [HD03, HD06, HD07]
HD1 = [HD13, HD16, HD17]
HD3 = [HD33, HD36, HD37]

fig_v3 = go.Figure(data=[go.Bar(name='No Enfermedad Cardiaca', x=Tal, y=HD0, marker=dict(color='palevioletred')),
                      go.Bar(name='Enfermedad Cardiaca Leve', x=Tal, y=HD1, marker=dict(color='lightpink')),
                      go.Bar(name='Enfermedad Cardiaca Severa', x=Tal, y=HD3, marker=dict(color='mistyrose'))])
fig_v3.update_layout(width=1000,barmode='group', yaxis=dict(title='Frecuencia'),  plot_bgcolor="rgba(255,255,255,255)"
                  , title='Efecto del Tipo de Talasemia en la Enfermedad Cardiaca', title_x=0.5, title_font_size=20,
                  legend_font_size = 16, xaxis = {'tickfont': {'size': 15}})
fig_v3.update_xaxes(range=[-0.5, 2.5], showline=True, linewidth=1, linecolor='black', mirror=True)
fig_v3.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)


# DASH -------------------------------------------------------------------------------------------------------------------
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Se definen los colores
colors = {'background': "#F4B5CC",'color': '#521383'}

app.layout = html.Div(children=[

    # Título
    html.H1('Sistema de Predicción de Enfermedad Cardíaca', style={'backgroundColor': colors['background'],
                                                                   'textAlign': 'center'}),
    html.Br(),
    dcc.Tabs(id = 'menupestanas', value = 'inicio', children = [

        # Tab 1
         dcc.Tab(label = 'Inicio',
                style={'backgroundColor': "#FCD7E5",'textAlign': 'center', 'font-size': '150%'},
                 value = 'inicio', children=[

                 html.Div([
                     html.Br(),
                     html.Br(),
                     html.Div(html.H6(
                         'Este sistema permite realizar predicciones del riesgo de sufrir una enfermedad cardíaca para determinar '
                         'el proceso adecuado a seguir, en busca del bienestar del paciente. Para esto, se tienen en cuenta los '
                         'siguientes parámetros:')),
                     html.Br(),
                     # Explicación de los parámetros utilizads
                     html.Div([
                         html.Div(html.H6(dcc.Markdown('''
                   * **Edad (Age):** Edad del paciente (años).
                   * **Glucosa (Fbs):** Nivel de glucosa en sangre en ayunas mayor a 120 mg/dL.
                   * **Colesterol (Chol):** Valor de colesterol total en sangre (mg/dL).
                   * **ST (Oldpeak):** Depresión del ST inducida por el ejercicio en relación con el reposo.
                   * **Angina (Exang):** Angina inducida por el ejercicio. 
                   * **Talasemia (Thal):** Tipo de talasemia.
                   ''')), style={'display': 'inline-block'}),
                         html.Div(html.Img(src=dash.get_asset_url("imagen1.png")),
                                  style={'height': '5%', 'display': 'inline-block'})])

                 ])
            ]),

        # Tab2
         dcc.Tab(label = 'Visualizaciones',
                style={'backgroundColor': "#FCD7E5",'textAlign': 'center', 'font-size': '150%'},
                 value = 'visuales', children=[

                 html.Div([
                     html.Br(),
                     html.Br(),
                     html.Div([
                         html.Div([
                             dcc.Graph(
                                 id='graph1',
                                 figure=fig_v1
                             ),
                         ], className='six columns'),

                         html.Div([
                             dcc.Graph(
                                 id='graph2',
                                 figure=fig_v2
                             ),
                         ], className='six columns'),
                     ], className='row'),

                     # New Div for all elements in the new 'row' of the page
                     html.Div([
                         dcc.Graph(
                             id='graph3',
                             figure=fig_v3
                         ),
                     ], className='row')
                 ])
             ]),

        # Tab 3
         dcc.Tab(label = 'Realizar Predicción', style={'backgroundColor': "#FCD7E5",'textAlign': 'center', 'font-size': '150%'},
                 value = 'prediccion', children=[

        html.Div([
                html.Br(),

                # Sección que indica la instrucción a seguir
                html.Div(html.H5('Seleccione los valores de los parámetros'),
                         style={'backgroundColor': "#FCD7E5",
                                'textAlign': 'center'}),
                html.Br(),
                # Se definen los parametros con los valores que pueden tomar:
                html.Div([
                    html.Div(html.H6("Edad (Age)", style={"color": "#A52555"})),
                    html.Div(
                        '''30: 29 a 39 años / 40: 40 a 49 años / 50: 50 a 59 años / 60: 60 a 69 años / 70: Mayor de 70 años'''),
                    html.Div([
                        dcc.Dropdown(
                            id='Edad',
                            options=[{'label': i, 'value': i} for i in [30, 40, 50, 60, 70]])],
                        style={'width': '35%', 'display': 'inline-block'}),

                    html.Div(html.H6("Glucosa (Fbs)", style={"color": "#A52555"})),
                    html.Div("0: No / 1: Sí"),
                    html.Div([
                        dcc.Dropdown(
                            id='Glucosa',
                            options=[{'label': i, 'value': i} for i in [0, 1]])],
                        style={'width': '35%', 'display': 'inline-block'})], style={'columnCount': 2}),

                html.Div([
                    html.Div(html.H6("Colesterol (Chol)", style={"color": "#A52555"})),
                    html.Div("0: menos de 200 / 1: Entre 200 y 239 / 2: Mayor o igual a 240"),
                    html.Div([
                        dcc.Dropdown(
                            id='Colesterol',
                            options=[{'label': i, 'value': i} for i in [0, 1, 2]])],
                        style={'width': '35%', 'display': 'inline-block'}),

                    html.Div(html.H6("ST (Oldpeak)", style={"color": "#A52555"})),
                    html.Div("0: Menos de 2 / 1: Entre 2 y 4 / 2: Mayor o igual a 4"),
                    html.Div([
                        dcc.Dropdown(
                            id='ST',
                            options=[{'label': i, 'value': i} for i in [0, 1, 2]])],
                        style={'width': '35%', 'display': 'inline-block'})], style={'columnCount': 2}),

                html.Div([
                    html.Div(html.H6("Angina (Exang)", style={"color": "#A52555"})),
                    html.Div("0: No / 1: Sí"),
                    html.Div([
                        dcc.Dropdown(
                            id='Ex',
                            options=[{'label': i, 'value': i} for i in [0, 1]])],
                        style={'width': '35%', 'display': 'inline-block'}),

                    html.Div(html.H6("Talasemia (Thal)", style={"color": "#A52555"})),
                    html.Div("3: Normal / 6: Defecto fijo / 7: Defecto reversible"),
                    html.Div([
                        dcc.Dropdown(
                            id='Talasemia',
                            options=[{'label': i, 'value': i} for i in [3, 6, 7]])],
                        style={'width': '35%', 'display': 'inline-block'})], style={'columnCount': 2}),

                # Se crea el botón
                html.Div([
                    html.Br(),
                    html.Br(),
                    html.Button('Realizar predicción', id='boton', n_clicks=0),
                    dcc.Interval(id='interval', interval=500)]),

                # Se crea la gráfica
                html.Div([
                    html.Br(),
                    html.Br(),
                    dcc.Graph(id='graficaProb')]),
                    ])
                 ])
            ])
        ])


# Función de Callback
@app.callback(
    Output('graficaProb', "figure"),
    [Input('boton', "n_clicks")],
    [State('Edad', 'value'),
     State('Glucosa', 'value'),
     State('Colesterol', 'value'),
     State('ST', 'value'),
     State('Ex', 'value'),
     State('Talasemia', 'value')],prevent_initial_call=True, suppress_callback_exceptions=True)

# Función para crear y actualizar la gráfica
def update_figure(n_clicks, age, Fbs, Chol, st, ex, tal):
    modelo1 = EstimacionModelos()[0]
    modelo2 = EstimacionModelos()[1]

    pred = modelo1.query(["HD"], evidence={"AGE": age, "FBS": Fbs, "CHOL": Chol, "OLDPEAK": st, "EXANG": ex, "THAL": tal})
    val1 = round(pred.values[0], 2)
    val2 = round(pred.values[1], 2)
    val3 = round(pred.values[2], 2)

    pred2 = modelo2.query(["HD"], evidence={"OLDPEAK": st, "EXANG": ex, "THAL": tal})
    val12 = round(pred2.values[0], 2)
    val22 = round(pred2.values[1], 2)
    val32 = round(pred2.values[2], 2)

    heart = ['No Heart Disease', 'Mild Heart Disease', 'Severe Heart Disease']

    # Se crea la gráfica de barras
    if math.isnan(val1) or math.isnan(val2) or math.isnan(val3):

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("No es posible calcular la probabilidad con los datos ingresados",
                                            "Modelo Estimado por Puntake K2"))
        fig.add_trace(go.Bar(x=heart, y=[val1, val2, val3],
                             text=[val1, val2, val3],
                             textposition='auto'), row=1, col=1)
        fig.update_traces(marker_color='lightpink', textfont_size=14)

        fig.add_trace(go.Bar(x=heart, y=[val12, val22, val32],
                             text=[val12, val22, val32],
                             textposition='auto'), row=1, col=2)
        fig.update_traces(marker_color='lightpink', textfont_size=14)

        fig.update_layout(width=1400, bargap=0.45,
                          plot_bgcolor="rgba(255,255,255,255)",
                          title_text='Probabilidad Estimada Enfermedad Cardiaca', title_x=0.5,
                          title_font_size=25,
                          showlegend=False)

        fig.update_xaxes(range=[-0.5, 2.5], showline=True, linewidth=1, linecolor='black', mirror=True,
                         tickfont=dict(size=15), title_text='Precisión : 0.64', row=1, col=1)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, tickfont=dict(size=15))


    elif math.isnan(val12) or math.isnan(val22) or math.isnan(val32):

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Modelo Estimado P1",
                                            "No es posible calcular la probabilidad con los datos ingresados"))
        fig.add_trace(go.Bar(x=heart, y=[val1, val2, val3],
                             text=[val1, val2, val3],
                             textposition='auto'), row=1, col=1)
        fig.update_traces(marker_color='lightpink', textfont_size=14)

        fig.add_trace(go.Bar(x=heart, y=[val12, val22, val32],
                             text=[val12, val22, val32],
                             textposition='auto'), row=1, col=2)
        fig.update_traces(marker_color='lightpink', textfont_size=14)

        fig.update_layout(width=1400, bargap=0.45,
                          plot_bgcolor="rgba(255,255,255,255)",
                          title_text='Probabilidad Estimada Enfermedad Cardiaca', title_x=0.5,
                          title_font_size=25,
                          showlegend=False)

        fig.update_xaxes(range=[-0.5, 2.5], showline=True, linewidth=1, linecolor='black', mirror=True,
                         tickfont=dict(size=15), title_text='Precisión : 0.63', row=1, col=1)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, tickfont=dict(size=15))


    else:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Modelo Estimado P1",
                                                            "Modelo Estimado por Puntake K2"))
        fig.add_trace(go.Bar(x=heart, y=[val1, val2, val3],
                             text=[val1, val2, val3],
                             textposition='auto'), row=1, col=1)
        fig.update_traces(marker_color='lightpink', textfont_size=14)

        fig.add_trace(go.Bar(x=heart, y=[val12, val22, val32],
                             text=[val12, val22, val32],
                             textposition='auto'), row=1, col=2)
        fig.update_traces(marker_color='lightpink', textfont_size=14)

        fig.update_layout(width=1400, bargap=0.45,
                          plot_bgcolor="rgba(255,255,255,255)",
                          title_text='Probabilidad Estimada Enfermedad Cardiaca', title_x=0.5,
                          title_font_size=25,
                          showlegend=False)

        fig.update_xaxes(range=[-0.5, 2.5], showline=True, linewidth=1, linecolor='black', mirror=True,
                         tickfont=dict(size=15), title_text='Precisión : 0.63', row=1, col=1)

        fig.update_xaxes(range=[-0.5, 2.5], showline=True, linewidth=1, linecolor='black', mirror=True,
                         tickfont=dict(size=15), title_text='Precisión : 0.64', row=1, col=2)

        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False, port=9878, host = "0.0.0.0")