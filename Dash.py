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
import plotly.graph_objs as go
import base64



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

df_bar = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df_bar, x="Fruit", y="Amount", color="City", barmode="group")

# Se definen los colores
colors = {'background': "#F4B5CC",'color': '#521383'}

app.layout = html.Div(children=[

    # Título
    html.H1('Sistema de Predicción de Enfermedad Cardíaca', style={'backgroundColor': colors['background'],
                                                                   'textAlign': 'center'}),
    html.Br(),
    dcc.Tabs(id = 'menupestanas',
                 value = 'inicio',
                 children = [

                     dcc.Tab(label = 'Inicio',
                            style={'backgroundColor': "#FCD7E5",'textAlign': 'center', 'font-size': '150%'},
                             value = 'inicio'),

                     dcc.Tab(label = 'Visualizaciones',
                            style={'backgroundColor': "#FCD7E5",'textAlign': 'center', 'font-size': '150%'},
                             value = 'visuales'),

                     dcc.Tab(label = 'Realizar Predicción',
                            style={'backgroundColor': "#FCD7E5",'textAlign': 'center', 'font-size': '150%'},
                             value = 'prediccion')

                 ]),

    html.Div(id = 'pestanas')

    ])


@app.callback(Output('pestanas', 'children'), Input('menupestanas', 'value'))
def tabFunction(tab):
    '''  '''

    return {'inicio' : tabInicio(tab), 'visuales' : tabVisual(tab), 'prediccion' : tabPred(tab)}[tab]


def tabInicio(tab):
    if (tab == 'inicio'):
        return html.Div([
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
            html.Div(html.Img(src=dash.get_asset_url("imagen1.png")), style={'height':'5%','display': 'inline-block'})])

        ])


def tabVisual(tab):
    if (tab == 'visuales'):
        return html.Div([
            html.Br(),
            html.Br(),
            html.Div([
            html.Div([
                html.H1(children='Hello Dash'),
                html.Div(children='''
                            Dash: A web application framework for Python.'''),
                dcc.Graph(
                    id='graph1',
                    figure=fig
                ),
            ], className='six columns'),

            html.Div([
                html.H1(children='Hello Dash'),
                html.Div(children='''
                            Dash: A web application framework for Python.'''),
                dcc.Graph(
                    id='graph2',
                    figure=fig
                ),
            ], className='six columns'),
        ], className='row'),
        # New Div for all elements in the new 'row' of the page
        html.Div([
            html.H1(children='Hello Dash'),

            html.Div(children='''
                        Dash: A web application framework for Python.
                    '''),

            dcc.Graph(
                id='graph3',
                figure=fig
            ),
        ], className='row')
        ])



def tabPred(tab):
    if (tab == 'prediccion'):
        return html.Div([
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
                html.Button('Realizar predicción', id='button', n_clicks=0),
                dcc.Interval(id='interval', interval=500)]),

            # Se crea la gráfica
            html.Div([
                html.Br(),
                html.Br(),
                dcc.Graph(id='graph-prob')]),

            html.Div(html.H5('Este sistema cuenta con una precisión del 70% y una cobertura del 65%'),
                     style={'backgroundColor': "#FC8BB3",
                            'textAlign': 'center'}),

        ])



if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False, port=9878)