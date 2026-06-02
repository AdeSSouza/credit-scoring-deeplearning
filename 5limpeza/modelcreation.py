#Pipeline de limpeza, tratamento, pre procesamento e criacao do modelo de Machine Learning

#importar bibliotecas
import pandas as pd
from datetime import datetime
import numpy as np
import random as python_random
import joblib

from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import tensorflow as tf

from utils import *
import const

#reproduzindo o mesmo resultado pra fins de comparacao
seed = 41
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)


#obtendo dados brutos
df = fetch_data_from_db(const.consulta_sql)


#convertendo tipos
df['idade'] = df['idade'].astype(int)
df['valorsolicitado'] = df['valorsolicitado'].astype(float)
df['valortotalbem'] = df['valortotalbem'].astype(float)


#Tratamento de Erros de Digitação especificamente a conluna profissoes em que há erros de digitacao
profissoes_validas = ['Advogado', 'Arquiteto', 'Cientista de Dados', 'Contador','Dentista','Empresário',
                                       'Engenheiro','Médico','Programador']
corrigir_erros_digitacao(df, 'profissao', profissoes_validas)


#Tratamento de nulos
df = substitui_nulos(df)


#Tratamento de Outliers, substitiu min,max pela mediana
df = tratar_outliers(df, 'tempoprofissao', 0, 70)
df = tratar_outliers(df, 'idade', 0, 110)


#Feature Engineering: interaction para criar novos atributos para avaliar o impacto na criacao do modelo
df['proporcaosolicitadototal'] = df['valorsolicitado'] / df['valortotalbem']
df['proporcaosolicitadototal'] = df['proporcaosolicitadototal'].astype(float)


# Dividindo os Dados em treino e teste, X variaveis independentes e y variavel dependente(classe)
X = df.drop('classe', axis=1)
y = df['classe']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=seed) # 20% dos dados


# Normalizando variaveis numericas X, boa pratica dividir em treino e teste antes de normalizar
X_test = save_scalers(X_test, ['tempoprofissao','renda','idade',
                               'dependentes','valorsolicitado','valortotalbem','proporcaosolicitadototal'])
X_train = save_scalers(X_train, ['tempoprofissao','renda','idade',
                                 'dependentes','valorsolicitado','valortotalbem','proporcaosolicitadototal'])


#Codificando dados categoricos em dados numericos, isso é feito de maneiras diferentes para X e y 
#por questoes de intuitividade das classes ruim=0 e bom=1
mapeamento = {'ruim': 0, 'bom': 1}
y_train = np.array([mapeamento[item] for item in y_train])
y_test = np.array([mapeamento[item] for item in y_test])
X_train = save_encoders(X_train, ['profissao', 'tiporesidencia', 
                                  'escolaridade','score','estadocivil','produto'])
X_test = save_encoders(X_test, ['profissao', 'tiporesidencia', 
                                'escolaridade','score','estadocivil','produto'])



# Selecionando Atributos com Random Forest baseado na importância para otimizar a entrada da Rede Neural.
model = RandomForestClassifier()
# Instancia o RFE
selector = RFE(model, n_features_to_select=10, step=1)
selector = selector.fit(X_train, y_train)
# Transforma os dados
X_train = selector.transform(X_train)
X_test = selector.transform(X_test)
joblib.dump(selector, './objects/selector.joblib')