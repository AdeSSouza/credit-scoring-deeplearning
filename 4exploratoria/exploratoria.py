# analise exploratoria

# import bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import const
from utils import *


# retornar dados do banco de dados utilizando a consulta SQL
df = fetch_data_from_db(const.consulta_sql)


# alguns ajustes de dados em formato correto
df['idade'] = df['idade'].astype(int)
df['valorsolicitado'] = df['valorsolicitado'].astype(float)
df['valortotalbem'] = df['valortotalbem'].astype(float)


#listas com variaveis categoricas e numericas separadas
variaveis_categoricas = ['profissao', 'tiporesidencia', 
                         'escolaridade', 'score', 'estadocivil', 'produto']
variaveis_numericas = ['tempoprofissao', 'renda', 'idade',
                        'dependentes', 'valorsolicitado', 'valortotalbem']


#grafico de barras com as variaveis categoricas
for coluna in variaveis_categoricas:
    df[coluna].value_counts().plot(kind='bar', figsize=(10, 6))
    plt.title(f'Distribuição de {coluna}')
    plt.ylabel('Contagem')
    plt.xlabel(coluna)
    plt.xticks(rotation=45)
    plt.show()


#boxplot e histograma com as variaveis numericas
for coluna in variaveis_numericas:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=coluna)
    plt.title(f'Boxplot de {coluna}')
    plt.show()

    df[coluna].hist(bins=20, figsize=(10, 6))
    plt.title(f'Histograma de {coluna}')
    plt.xlabel(coluna)
    plt.ylabel('Frequência')
    plt.show()


    #resumo estatistico
    print(f'Resumo estatístico de :\n', df[coluna].describe(), '\n')


#verificar existencia de valores nulos
nulos_por_coluna = df.isnull().sum()
print(nulos_por_coluna)