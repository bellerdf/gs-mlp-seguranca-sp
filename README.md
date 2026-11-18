# Classificação de Ocorrências Policiais em São Paulo com MLP

Projeto da Global Solution — Aprendizado Profundo com Redes Neurais Convolucionais (FIAP).

**Integrantes da Equipe:**
* Isabela Marques de Oliveira (RM: 567230)
* Isabelle Ramos de Filippis (RM: 566783)
* Samy Tamires de Sousa Cruz (RM: 566674)

**Aplicação rodando no Streamlit:** https://gs-mlp-seguranca-sp-qsit7kr2bcmqrtbddayvyi.streamlit.app/

## 1. Sobre o projeto
O objetivo aqui foi construir, treinar, avaliar e comparar três arquiteturas diferentes de redes neurais Multilayer Perceptron (MLP) para classificar ocorrências registradas na capital de São Paulo entre **Roubo (1)** e **Furto (0)**. 

Vale destacar que o modelo não prevê se um crime vai acontecer no futuro. Ele analisa as características de uma ocorrência (já registrada ou simulada no formulário) para indicar se o padrão daquele registro é mais compatível com roubo ou furto.

## 2. Base de dados e pré-processamento
Os dados vieram da base pública de celulares subtraídos da Secretaria da Segurança Pública de São Paulo (SSP-SP), referente ao ano de 2026 (`CelularesSubtraidos_2026.xlsx`, aba `CELULAR_2026`), disponível no Portal de Estatísticas: https://www.ssp.sp.gov.br/estatistica/consultas.

* **Filtragem e limpeza:** Dos 256.076 registros originais, filtramos apenas os fatos ocorridos no município de São Paulo (`NOME_MUNICIPIO_CIRC`: 151.170 registros) e removemos boletins duplicados pela chave composta (`ANO_BO`, `NUM_BO`, `ID_DELEGACIA`), chegando a **130.851 ocorrências únicas**.
* **Amostra e divisão:** Para viabilizar o treino, utilizamos um recorte estratificado de **18.000 registros** (65,04% de furto e 34,96% de roubo), divididos em **70% para treino** (12.600), **15% para validação** (2.700) e **15% para teste** (2.700).
* **Features e prevenção de Data Leakage:** Utilizamos `BAIRRO_AGRUPADO`, `PERIODO_DIA`, `DIA_SEMANA`, `TIPOLOCAL_AGRUPADO`, `HORA_NUM` e `FIM_DE_SEMANA`. Após a imputação de dados ausentes, codificação categórica (*OneHotEncoding*) e escalonamento numérico (*StandardScaler*), a rede recebeu **53 features de entrada**. Colunas como `RUBRICA` e `DESC_LEI` foram excluídas das entradas para evitar vazamento de informação (*data leakage*).

## 3. Configuração do treinamento e resultados (Validação)
O treinamento utilizou função de perda `binary_crossentropy`, otimizador `Adam` (`lr=0.001`), `batch_size=64`, limite de `50 épocas` com `EarlyStopping` (`patience=8`) e balanceamento de classes via `ClassWeight`.

| Arquitetura | Camadas Ocultas | Ajustes | Acurácia | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Arquitetura 2 (Escolhida)** | 3 (128 - 64 - 32, ReLU) | EarlyStopping + ClassWeight | **69,04%** (0.6904) | 54,93% (0.5493) | 63,77% (0.6377) | **0.5902** |
| **Arquitetura 1 (Baseline)** | 1 (32, ReLU) | EarlyStopping + ClassWeight | 68,41% (0.6841) | 54,02% (0.5402) | **64,72%** (0.6472) | 0.5889 |
| **Arquitetura 3 (Regularizada)** | 2 (64 - 32, ReLU) | Dropout (0.3/0.2) + L2 + BatchNorm | 67,81% (0.6781) | **69,04%** (0.6904) | 14,41% (0.1441) | 0.2384 |

### Desempenho Final no Conjunto de Teste
A **Arquitetura 2 (Profunda 128-64-32)** atingiu o melhor equilíbrio de *F1-Score* na validação (evitando a queda severa de *Recall* observada na Configuração 3) e foi escolhida para integração na aplicação web. Avaliada no conjunto de teste isolado (2.700 amostras), obteve:
* **Acurácia:** 69,81%
* **Precision:** 56,04%
* **Recall:** 63,35%
* **F1-Score:** 0,5947

## 4. Como rodar na sua máquina

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Execute a aplicação no Streamlit:
```bash
streamlit run app.py
```
