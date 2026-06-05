#funcoes que serao utilizadas durante o pre processamento e criacao do modelo

#importar bibliotecas
from fuzzywuzzy import process #logica difusa
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder # normalizar dados e variaveis categoricas
import joblib # salvar os objetos
import yaml # ler arquivo de configuracoes
import psycopg2 # conectar ao banco de dados
import const #arquivo de constante


#recuperar os dados atualizados do BD
def fetch_data_from_db(sql_query):
    try:
        with open('config.yaml', 'r') as file: # arquivo de configuracoes de conexao
            config = yaml.safe_load(file)

        con = psycopg2.connect(
            dbname=config['database_config']['dbname'], 
            user=config['database_config']['user'], 
            password=config['database_config']['password'], 
            host=config['database_config']['host']
        )

        # executar a consulta sql recebida como parametro
        cursor = con.cursor()
        cursor.execute(sql_query)

        # retornar todos os registros
        df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

    #liberar os recursos
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'con' in locals():
            con.close()

    #retorna o date frame
    return df

#corrigir erros de digitacao com logica difusa
def corrigir_erros_digitacao(df, coluna, lista_valida):
    for idx, linha in df.iterrows():
        valor = linha[coluna]
        
        if pd.isnull(valor):
            continue
            
        valor_str = str(valor).strip()
        
        if valor_str not in lista_valida:
            resultado = process.extractOne(valor_str, lista_valida)
            if resultado:
                # Pega estritamente apenas o texto corrigido da tupla
                df.at[idx, coluna] = resultado[0]


#substituir valores nulos
def substitui_nulos(df):
    for coluna in df.columns:
        # Verifica se a coluna é do tipo numérico
        if pd.api.types.is_numeric_dtype(df[coluna]):
            mediana = df[coluna].median()
            df[coluna] = df[coluna].fillna(mediana)
        else:
            # Para colunas de texto/object, calcula a moda se houver valores
            if not df[coluna].dropna().empty:
                moda = df[coluna].mode()[0]
                df[coluna] = df[coluna].fillna(moda)
                
    return df


#tratar outliers numericos com mediana
def tratar_outliers(df, coluna, minimo, maximo):
    mediana = df[(df[coluna] >= minimo) & (df[coluna] <= maximo)][coluna].median()
    df[coluna] = df[coluna].apply(lambda x: mediana if x < minimo or x > maximo else x)
    return df


#normalizar dados numericos
def save_scalers(df, nome_colunas):
    for nome_coluna in nome_colunas:
        scaler = StandardScaler()
        df[nome_coluna] = scaler.fit_transform(df[[nome_coluna]])
        joblib.dump(scaler, f"./objects/scaler{nome_coluna}.joblib")

    return df


#decodificar variaveis categoricas
def save_encoders(df, nome_colunas):
    for nome_coluna in nome_colunas:
        label_encoder = LabelEncoder()
        df[nome_coluna] = label_encoder.fit_transform(df[nome_coluna])
        joblib.dump(label_encoder, f"./objects/labelencoder{nome_coluna}.joblib")

    return df

#carregar scalers para API
def load_scalers(df, nome_colunas):
    for nome_coluna in nome_colunas:
        nome_arquivo_scaler = f"./objects/scaler{nome_coluna}.joblib"
        scaler = joblib.load(nome_arquivo_scaler)
        df[nome_coluna] = scaler.transform(df[[nome_coluna]])
    return df

#carregar enconders para API  
def load_encoders(df, nome_colunas):
    for nome_coluna in nome_colunas:
        nome_arquivo_encoder = f"./objects/labelencoder{nome_coluna}.joblib"
        label_enconder = joblib.load(nome_arquivo_encoder)
        df[nome_coluna] = label_enconder.transform(df[[nome_coluna]])   
    return df
