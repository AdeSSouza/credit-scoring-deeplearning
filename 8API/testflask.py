
#importando bibliotecas

import requests
import yaml

#lendo a configuracao
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    url = config['url_api']['url']

#instancias para fazer o teste com a API
dados_novos = {
    'profissao': ['Advogado','Médico','Dentista','Contador'],
    'tempoprofissao': [39, 37, 16, 0],
    'renda': [20860.0, 5000, 20000, 7000],
    'tiporesidencia': ['Alugada','Própria','Própria','Alugada'],
    'escolaridade': ['Ens.Fundamental','PósouMais','Superior','Ens.Fundamental'],
    'score': ['Baixo','Baixo','MuitoBom','MuitoBom'],
    'idade': [36, 25, 19, 24],
    'dependentes': [0, 0, 4, 2],
    'estadocivil': ['Víuvo','Casado','Casado','Solteiro'],
    'produto': ['DoubleDuty','SpeedFury','ElegantCruise','TrailConqueror'],
    'valorsolicitado': [139244.0, 100000, 50000, 200000],
    'valortotalbem': [320000.0, 200000, 200000, 300000],
    'proporcaosolicitadototal': [2.2, 50, 200, 40]
}

#chamada a API
response = requests.post(url, json=dados_novos)

#verificando resposta da API
if response.status_code == 200:
    print("Previsões recebidas:")
    predictions = response.json()
    print(predictions)
else:
    print("Erro ao fazer a previsão:", response.status_code)
                