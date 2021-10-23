# Importando as bibliotecas

import numpy as np
import pandas as pd
import missingno as miss
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly_express as px
import seaborn as sn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

# Usaremos o banco de dados heart failure para esse exemplo. Por ser uma base sobre doenças vamos focar em melhorar
# as respostas positivas
# Abrindo os dados!
#Os dados usados nesse notebook foram retirados do kaggle, segue o link:
#https://www.kaggle.com/andrewmvd/heart-failure-clinical-data

dados = pd.read_csv(r"C:\Users\anacl\Documents\heart_failure.csv")
print(dados)

# Primeiramente vamos analisar a proporção entre os outcomes, esse passo é muito importante porque a proporção das categorias
# quando desbalanceadas os modelos, por exemplo uma diferença de 40% ou mais entre as variaveis, cuidados maiores devem ser tomados.
#Quando as classes são balanceadas, ou seja, estão mais ou menos em mesma quantidade, os modelos costumam trabalhar bem melhor
#Faremos um grafico de pizza animado sobre a coluna de outcome, no nosso caso a coluna 'DEATH_EVENT'

pizza = px.pie(dados, names='DEATH_EVENT', title= 'Proporção entre os desfechos')
pizza.show()

# Com isso podemos observar que a diferença é um pouco desbalanceada mas ainda não chegamos na 'dangerous zone'
# Outra coisa importante é observar se há dados faltantes na base de dados, e se isso ocorrer substituir os dados pelas
# medias ou medianas de cada coluna

miss.matrix(dados)

# Como os dados, felizmente, estão todos preenchidos podemos pular essa etapa. Proximo passo que também é muito importante
# é ver quais as variaveis vão entrar no nosso modelo, com isso é interessante ver como elas estão escaladas e se possuem
# muitos outliers ou nenhum. Como são muitas variaveis vou mostrar apenas algumas aqui.
# Obs: vamos fazer essa analise por curiosidade já que esse metodo os outliers não dao muita e=interferencia no resultado
# como seria por exemplo com a regressão.

box = px.box(dados, y = 'creatinine_phosphokinase')
box.show()
# creatinine_phosphokinase possui muitos outliers.

box = px.box(dados, y = 'ejection_fraction')
box.show()

## Agora a variavel ejection_fraction, parece ser mais confiavel com menos outliers.
## Vamos ver se tem alguma correlação entre as colunas

correlation = dados.corr()
plot = sn.heatmap(correlation, annot = True, fmt=".1f", linewidths=.6)
plot

# Traduzindo o grafico...
# ρ = 0,9 a 1 (positivo ou negativo): correlação muito forte;
# ρ = 0,7 a 09 (positivo ou negativo): correlação forte;
# ρ = 0,5 a 0,7 (positivo ou negativo): correlação moderada;
# ρ = 0,3 a 0,5 (positivo ou negativo): correlação fraca;
# ρ = 0 a 0,3 (positivo ou negativo): não possui correlação

## Podemos notar que a maioria não tem correlação ou possui uma muito fraca.

# Agora vamos separar as variaveis preditoras e de desfecho. No nosso caso, iremos usar como preditoras todas exceto time,
# smoking, sex, anaemia, diabetes e death event, sendo que death event vai ser nossa variavel de desfecho.

# Função y = f(x)

x = dados.drop(columns=['DEATH_EVENT','time','smoking','anaemia', 'diabetes', 'sex'])
y = dados['DEATH_EVENT']

# Vamos dividir a amostra em 75% e 25%, metodo mais convencional.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

# Aplicando o metódo de KNN
# Pequena explicação de como funciona o algoritmo de KNN. Ele basicamente pega as obervações proximas que são determinadas pelo
# tamanho de K, e então vota pela categoria que possui maior similaridade 

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x,y) #treinando o modelo
y_predict = knn.predict(x_test)

#Criando modelo "bobinho"

dummymodel = DummyClassifier (strategy = "stratified")
dummymodel.fit(x_train, y_train)
y_pred_dummy = dummymodel.predict(x_test)

#Avaliando o Modelo

acuracia = accuracy_score(y_test, y_predict)

acuracia_dummy = accuracy_score(y_test, y_pred_dummy)

plot_confusion_matrix(knn, x_test, y_test, normalize= 'pred')
plt.show()

#usando Fine Tunning para melhorar a precisão, achando qual é o melhor tamanho de K

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