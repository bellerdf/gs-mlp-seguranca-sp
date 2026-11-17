# Classificação de Ocorrências Policiais em São Paulo com MLP

Projeto da Global Solution — Aprendizado Profundo com Redes Neurais Convolucionais (FIAP).

**Aplicação rodando no Streamlit:** https://gs-mlp-seguranca-sp-qsit7kr2bcmqrtbddayvyi.streamlit.app/

## Sobre o projeto
O objetivo aqui foi treinar e comparar três arquiteturas diferentes de redes neurais Multilayer Perceptron (MLP) para classificar ocorrências registradas na capital de São Paulo entre **Roubo (1)** e **Furto (0)**. 

Vale destacar que o modelo não prevê se um crime vai acontecer no futuro. Ele analisa as características de uma ocorrência (já registrada ou simulada no formulário) para indicar se o padrão daquele registro bate mais com roubo ou furto.

## Base de dados e pré-processamento
Os dados vieram da base pública de celulares subtraídos da Secretaria da Segurança Pública de São Paulo (SSP-SP), referente ao ano de 2026 (`CelularesSubtraidos_2026.xlsx`, aba `CELULAR_2026`)[cite: 16].

* **Filtragem e limpeza:** Dos 256.076 registros originais, separei apenas os ocorridos no município de São Paulo (151.170) e removi boletins duplicados, chegando a 130.851 ocorrências únicas[cite: 16].
* **Amostra:** Para viabilizar o treino, utilizei um recorte de **18.000 registros** (65,04% de furto e 34,96% de roubo), divididos em 70% para treino (12.600), 15% para validação (2.700) e 15% para teste (2.700).
* **Features:** Usei `BAIRRO_AGRUPADO`, `PERIODO_DIA`, `DIA_SEMANA`, `TIPOLOCAL_AGRUPADO`, `HORA_NUM` e `FIM_DE_SEMANA`. Depois do *OneHotEncoding* e do *StandardScaler*, ficaram 53 entradas[cite: 17]. Colunas como `RUBRICA` e `DESC_LEI` foram cortadas antes do treino para não vazar a resposta para a rede (*data leakage*)[cite: 17].

## Resultados dos experimentos (Validação)

| Arquitetura | Camadas Ocultas | Ajustes | Acurácia | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Arquitetura 2 (Escolhida)** | 3 (128 - 64 - 32, ReLU) | EarlyStopping + ClassWeight | **0.6904** | 0.5493 | 0.6377 | **0.5902** |
| **Arquitetura 1 (Baseline)** | 1 (32, ReLU) | EarlyStopping + ClassWeight | 0.6841 | 0.5402 | 0.6472 | 0.5889 |
| **Arquitetura 3 (Regularizada)** | 2 (64 - 32, ReLU) | Dropout (0.3/0.2) + L2 + BatchNorm | 0.6781 | 0.6904 | 0.1441 | 0.2384 |

A **Arquitetura 2** teve o melhor equilíbrio entre *precision* e *recall* na validação e foi escolhida para ir para a aplicação web[cite: 17]. Rodando ela no conjunto de teste final (2.700 registros), os números foram: **69,81% de acurácia**, **56,04% de precision**, **63,35% de recall** e **0,5947 de F1-score**.

## Como rodar na sua máquina

Instale as dependências:
```bash
pip install -r requirements.txt
