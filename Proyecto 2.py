# Alejandra Erazo / Juliana Mendoza
# Modelo

## Médicos que quieran incorporar un apoyo de analítica de datos en el proceso de evaluación de pacientes y
# la toma de decisiones asociada (solicitud de exámenes, chequeos y otros procedimientos).

# Se importan las librerias
import pandas as pd
import numpy as np
import seaborn as sns
from pgmpy.models import BayesianNetwork, BayesianModel
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator, PC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score, BicScore
import networkx as nx

##
# Se leen los datos
data =  pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data', header=None)
names= ["age","sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope","ca", "thal", "num"]
data.columns = names
data['ca'] = pd.to_numeric(data['ca'], errors='coerce')
data['thal'] = pd.to_numeric(data['thal'], errors='coerce')
data = data.astype(float)
##
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
##

#
#Se dividen los datos

DataEntrenamiento = data[0:250,]
DataValidacion = data[250:,]

#Se define el modelo
##
# Se define la red bayesiana
modelo_HD = BayesianNetwork([("AGE", "CHOL"), ("FBS", "CHOL"), ("CHOL", "HD"), ("THAL", "HD"), ("HD", "EXANG"),
                         ("HD", "OLDPEAK")])

# Se definen las muestras
info = np.zeros((250,7))
columnas = [0, 4, 5, 8, 9, 12, 13]
nombres = ["AGE", "CHOL", "FBS", "EXANG", "OLDPEAK", "THAL", "HD"]
for i in range(len(columnas)):
    info[:,i] = DataEntrenamiento[:,columnas[i]]
muestras = pd.DataFrame(info, columns = nombres)

#Estimación de las CPDs
modelo_HD.fit(data = muestras, estimator = MaximumLikelihoodEstimator)
for i in modelo_HD.nodes():
    print("CPD ", i,"\n", modelo_HD.get_cpds(i))

#Se imprime completa la CDP HD
for i in range(len(modelo_HD.get_cpds("HD").values)):
    print(modelo_HD.get_cpds("HD").values[i])

# Se realiza la eliminación de variables
infer = VariableElimination(modelo_HD)
##
# Se definen valores para los parámetros para la predicción
age = 60
Fbs = 1
Chol = 2
st = 1
ex = 1
tal = 7

# Se realiza la predicción teniendo en cuenta la evidencia
posterior_p = infer.query(["HD"], evidence={"AGE": age, "FBS": Fbs, "CHOL": Chol,  "OLDPEAK": st, "EXANG": ex, "THAL": tal})
print(posterior_p)
valores = posterior_p.values
print(valores)

#######

Val = np.zeros((47,7))
columnas = [0, 4, 5, 8, 9, 12, 13]
nombres = ["AGE", "CHOL", "FBS", "EXANG", "OLDPEAK", "THAL", "HD"]
for i in range(len(columnas)):
    Val[:,i] = DataValidacion[:,columnas[i]]

##
predicciones = []
anotaciones = []

for i in range(0, len(Val)):
    age, chol, Fbs, ex, st, tal = Val[i, 0], Val[i, 1], Val[i, 2], Val[i, 3], \
                                  Val[i, 4], Val[i, 5]

    anotaciones = np.append(anotaciones, Val[i, 6])

    posterior_p = infer.query(["HD"],evidence={"AGE": age, "FBS": Fbs, "CHOL": Chol,
                                               "OLDPEAK": st, "EXANG": ex, "THAL": tal})

    probabilidades = posterior_p.values

    if np.isnan(probabilidades).any():
        anotaciones = np.delete(anotaciones, i, axis=0)

    maximo = np.max(probabilidades)
    for j in range(0, len(probabilidades)):
        if probabilidades[j] == maximo:
            posicion = j
            if posicion == 2:
                predicciones = np.append(predicciones, 3)
            else:
                predicciones = np.append(predicciones, posicion)
##
matriz = confusion_matrix(anotaciones, predicciones)

cm_display = ConfusionMatrixDisplay(confusion_matrix = matriz,
                                    display_labels = ['No EC', 'EC Leve','EC Severa'])


cm_display.plot(cmap = 'PuRd', colorbar = True)
plt.title('Matriz de Confusión para Predicción de Enfermedad Cardiaca (EC) \n')
plt.ylabel('Anotaciones')
plt.xlabel('Predicciones')
plt.tight_layout()
plt.show()

precision = precision_score(anotaciones, predicciones, average = 'micro')

cobertura = recall_score(anotaciones, predicciones, average = 'micro')

f1 = f1_score(anotaciones, predicciones, average = 'micro')
##
#Estimación por Puntaje
nombres = ["AGE", "CHOL", "FBS", "EXANG", "OLDPEAK", "THAL", "HD"]

DEntr = np.zeros((250,7))
columnas = [0, 4, 5, 8, 9, 12, 13]

for i in range(len(columnas)):
    DEntr[:,i] = DataEntrenamiento[:,columnas[i]]


DFEntrenamiento = pd.DataFrame(DEntr, columns = nombres)
##
scoring_method = K2Score(data=DFEntrenamiento)
esth = HillClimbSearch(data = DFEntrenamiento)
estimated_modelh = esth.estimate(
    scoring_method=scoring_method, max_indegree = 8, max_iter=int(1e4))

print(estimated_modelh)
print(estimated_modelh.nodes())
print(estimated_modelh.edges())

graph = nx.DiGraph()
graph.add_nodes_from(estimated_modelh.nodes())
graph.add_edges_from(estimated_modelh.edges())
plt.figure(figsize = (5,5))
pos = {'EXANG': (1.9, 0.3), 'THAL': (0.2, 1), 'FBS': (1.02, 0.3), 'CHOL': (1.9, 1), 'OLDPEAK': (1.9, 1.7), 'AGE': (1.02, 1.7), 'HD': (1, 1)}
nx.draw(graph, pos = pos, with_labels = True, node_color = 'pink', node_size = 2300, font_size = 10, arrowsize = 20)

##

DFValidacion = pd.DataFrame(Val, columns = nombres)
modeloK2 = BayesianNetwork()
edges = estimated_modelh.edges()

modeloK2.add_edges_from(edges)
modeloK2.fit(data = DFEntrenamiento, estimator = MaximumLikelihoodEstimator)

for i in modeloK2.nodes():
    print("CPD ", i,"\n", modeloK2.get_cpds(i))

#Se imprime completa la CDP HD
for i in range(len(modeloK2.get_cpds("HD").values)):
    print(modeloK2.get_cpds("HD").values[i])

# Se realiza la eliminación de variables
K2Mod = VariableElimination(modeloK2)
##

prediccionesK2 = []
anotacionesK2 = []

for i in range(0, len(Val)):
    exk2, stk2, talk2 = Val[i, 3],Val[i, 4], Val[i, 5]

    anotacionesK2 = np.append(anotacionesK2, Val[i, 6])

    posterior_pk2 = K2Mod.query(["HD"],evidence={"OLDPEAK": stk2, "EXANG": exk2, "THAL": talk2})

    probabilidadesk2 = posterior_pk2.values

    if np.isnan(probabilidadesk2).any():
        anotacionesK2 = np.delete(anotacionesK2, i, axis=0)

    maximok2 = np.max(probabilidadesk2)
    for j in range(0, len(probabilidadesk2)):
        if probabilidadesk2[j] == maximok2:
            posicionk2 = j
            if posicionk2 == 2:
                prediccionesK2 = np.append(prediccionesK2, 3)
            else:
                prediccionesK2 = np.append(prediccionesK2, posicionk2)
##
matrizK2 = confusion_matrix(anotacionesK2, prediccionesK2)

cm_displayK2 = ConfusionMatrixDisplay(confusion_matrix = matrizK2,
                                    display_labels = ['No EC', 'EC Leve','EC Severa'])


cm_display.plot(cmap = 'PuRd', colorbar = True)
plt.title('Matriz de Confusión para Predicción de Enfermedad Cardiaca (EC) \n con Estimación por Puntaje K2 \n')
plt.ylabel('Anotaciones')
plt.xlabel('Predicciones')
plt.tight_layout()
plt.show()

precisionK2 = precision_score(anotacionesK2, prediccionesK2, average = 'micro')

coberturaK2 = recall_score(anotacionesK2, prediccionesK2, average = 'micro')

f1K2 = f1_score(anotacionesK2, prediccionesK2, average = 'micro')