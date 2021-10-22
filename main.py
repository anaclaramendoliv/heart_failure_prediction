# Importando as bibliotecas

import numpy as np
import pandas as pd
import missingno as miss
import matplotlib.pyplot as plt
import plotly.graph_objects as go


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


# Abrindo os dados!
#Os dados usados nesse notebook foram retirados do kaggle, segue o link:
#https://www.kaggle.com/andrewmvd/heart-failure-clinical-data

dados = pd.read_csv(r"C:\Users\anacl\Documents\heart_failure.csv")
dados

# Primeiramente vamos analisar a proporção entre os outcomes
#Faremos um grafico de pizza animado sobre a coluna de outcome, no nosso caso a coluna 'DEATH_EVENT'



# Por curiosidade vamos abrir as estatisticas da base de dados
dados.describe()

#Observar se há dados faltantes na base de dados, se isso ocorre substituir os dados pelas medias ou medianas de cada coluna


miss.matrix(dados)

#Como os dados, felizmente, estão todos preenchidos podemos pular essa etapa.Agora vamos separar as variaveis preditoras e de desfecho. No nosso caso, iremos usar como preditoras todas exceto time, smoking, sex anaemia, diabetes e death event, sendo que death event vai ser nossa variavel de desfecho.

#Função y = f(x)

dados.columns


x = dados.drop(columns=['DEATH_EVENT','time','smoking','anaemia', 'diabetes', 'sex'])
y = dados['DEATH_EVENT']
x


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

#Aplicando o metódo de KNN

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x,y) #treinando o modelo
y_predict = knn.predict(x_test)

#Criando modelo "bobinho"

dummymodel = DummyClassifier (strategy = "stratified")
dummymodel.fit(x_train, y_train)
y_pred_dummy = dummymodel.predict(x_test)

#Avaliando o Modelo


acuracia = accuracy_score(y_test, y_predict)
acuracia

acuracia_dummy = accuracy_score(y_test, y_pred_dummy)
acuracia_dummy

plot_confusion_matrix(knn, x_test, y_test, normalize= 'pred')
plt.show()

#usando Fine Tunning para melhorar a acuracia
#*texto em itálico*

lista_k = []
lista_tp = []

for k in range (3, 51, 2):

  knn = KNeighborsClassifier(n_neighbors = k)
  knn.fit(x,y) #treinando o modelo
  y_predict = knn.predict(x_test)
  tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
  lista_k.append(k)
  lista_tp.append(tp)


fig = go.Figure(data = go.Scatter(x = lista_k, y = lista_tp))
fig.show()

##Com isso podemos observar que o melhor valor de K é realmente 3 o valor que escolhemos anteriormente aleatoriamente