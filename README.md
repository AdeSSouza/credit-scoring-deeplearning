# Implementação Prática de Modelos Preditivos para Concessão de Crédito

> Abordagem técnica utilizando SQL, Machine Learning e Redes Neurais para avaliação de risco no setor automotivo.

Este projeto desenvolve uma solução para analisar dados históricos de clientes, identificar padrões de inadimplência e automatizar a tomada de decisão para concessão de crédito automotivo de forma segura e eficiente.

---

## 🛠️ Stack Técnica e Ferramentas

* **Engenharia de Dados:** SQL (Joins, Aggregations), PostgreSQL e `psycopg2-binary`.
* **Processamento e Estatística:** Python, Pandas, NumPy e Estatística Descritiva.
* **Engenharia de Atributos:** Scikit-Learn (seleção de features com Random Forest, codificação de categorias com Label/One-Hot Encoding, Interactions).
* **Deep Learning (Modelagem):** Keras e TensorFlow (Redes Neurais Artificiais).
* **Interpretabilidade (XAI):** LIME (Local Interpretable Model-agnostic Explanations).
* **Infraestrutura e Serviços:** Flask (API REST), Streamlit (Interface de Usuário) e AWS EC2.

---

## 🎯 Requisitos e Métricas de Sucesso (KPIs)

O modelo foi desenvolvido seguindo critérios estritos de viabilidade econômica e tolerância ao risco:
* **Métricas Alvo:** Precision esperada de ≥ 80% (mínima de 70%) e Recall esperado de ≥ 75% (mínima de 70%).
* **Gerenciamento de Risco:** Implementação de limiar de decisão (Classification Threshold) flexível a partir de 50%, permitindo o ajuste fino entre apetite de risco e taxa de aprovação.
* **Deploy e Produção:** Disponibilização do modelo final através de uma API REST integrada a uma interface local em Streamlit para consumo em tempo real.
* **Hospedagem:** Estrutura preparada para provisionamento em ambiente de nuvem utilizando instâncias AWS EC2.

---

## ⚙️ Pipeline de Engenharia e Pré-processamento de Dados

O tratamento da base de dados bruta foi estruturado para evitar distorções nos dados, impedir o vazamento de dados (*data leakage*) e otimizar o aprendizado dos modelos:
1. **Saneamento e Tipagem:** Conversão e coerção de tipos de dados estruturados com Pandas.
2. **Tratamento de Inconsistências Textuais:** Aplicação de algoritmos de Lógica Difusa (*Fuzzy Logic* via FuzzyWuzzy) para correção automatizada de erros de digitação em campos abertos antes das agregações.
3. **Tratamento de Dados Ausentes:** Imputação utilizando a Moda para variáveis categóricas e a Mediana para variáveis numéricas (evitando distorções por outliers).
4. **Tratamento de Outliers:** Identificação e contenção de anomalias através de análises baseadas em Desvio Padrão e Intervalo Interquartil (IQR).
5. **Engenharia de Atributos:** Criação de novos recursos matemáticos por meio de interações (*interactions*) entre variáveis para capturar relações não-lineares.
6. **Validação Cruzada:** Divisão estrita dos dados em conjuntos de treino e teste utilizando o Scikit-Learn antes de qualquer transformação de escala.
7. **Escalonamento de Recursos:** Normalização e padronização estatística de variáveis numéricas aplicando Z-score e redimensionamento Min-Max.
8. **Codificação Categórica:** Transformação de dados qualitativos em numéricos utilizando abordagens de Label Encoding e One-Hot Encoding conforme a cardinalidade do atributo.
9. **Seleção de Atributos:** Redução de dimensionalidade baseada na importância dos recursos (*feature importance*) extraída via algoritmo Random Forest para otimizar a entrada da Rede Neural.

---

## 📂 Organização das Pastas (Evolução do Repositório)

Este repositório foi estruturado cronologicamente para registrar a evolução e o desenvolvimento de cada fase do pipeline do projeto:

* **`5limpeza/`** → Desenvolvimento e consolidação do pipeline de higienização de dados brutos do PostgreSQL, integrando a lógica difusa e regras de imputação estatística.
* **`6modelo/`** → Definição da arquitetura da Rede Neural no Keras/TensorFlow, rotinas de treinamento, otimização e avaliação das métricas de classificação.
* **`7explicabilidade/`** → Implementação do script `xai.py` utilizando o framework **LIME** para auditoria de previsões de caixas-pretas e geração do relatório visual `lime_explanation.html`.
* **`8API/`** → Conclusão do ecossistema de serviço contendo o backend da API REST (`api.py` com Flask) e a interface gráfica do usuário (`webapp.py` com Streamlit).

---

## 🚀 Como Executar Localmente

### 1. Configuração do Ambiente Virtual
```bash
# Ative seu ambiente virtual (Exemplo no Windows)
.venv\Scripts\Activate.ps1

# Instale as dependências unificadas do projeto
pip install -r requirements.txt
```

### 2. Inicialização dos Serviços
Abra dois terminais na pasta raiz do projeto e execute:

* **Terminal 1 (Backend - API Flask):**
  ```bash
  python 8API/api.py
  ```
* **Terminal 2 (Frontend - Streamlit):**
  ```bash
  streamlit run 8API/webapp.py
  ```

---
*Projeto desenvolvido com foco prático no domínio técnico de ferramentas de mercado para Ciência de Dados.*
