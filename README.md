# Segurança Pública Inteligente em São Paulo: Classificação de Ocorrências com MLP

Projeto desenvolvido para a Global Solution da disciplina Aprendizado Profundo com Redes Neurais Convolucionais (FIAP).

## Link da Aplicação Web Funcional
Acesse a Aplicação Web aqui: https://gs-mlp-seguranca-sp-qsit7kr2bcmqrtbddayvyi.streamlit.app/

## 1. Objetivo do Projeto
Construir, treinar, avaliar e comparar três arquiteturas de redes neurais Multilayer Perceptron (MLP) para classificar ocorrências policiais registradas na cidade de São Paulo entre Roubo (1) e Furto (0). O modelo analisa uma ocorrência registrada ou hipotética e estima sua compatibilidade com roubo ou furto, sem prever o risco futuro de um crime acontecer.

## 2. Fonte de Dados e Procedimento de Obtenção
- **Fonte Pública:** Secretaria da Segurança Pública do Estado de São Paulo (SSP-SP) - Portal de Estatísticas ([https://www.ssp.sp.gov.br/estatistica/consultas](https://www.ssp.sp.gov.br/estatistica/consultas)).
- **Base Escolhida:** CelularesSubtraidos_2026.xlsx (Aba CELULAR_2026).
- **Recorte e Dimensionamento:**
  - Registros brutos na base: 256.076
  - Filtro geográfico para fatos ocorridos no município de São Paulo (Capital) via NOME_MUNICIPIO_CIRC: 151.170
  - Remoção de duplicidades por Boletim de Ocorrência (ANO_BO, NUM_BO, ID_DELEGACIA): 130.851 BOs únicos
  - Amostra final dimensionada: 18.000 observações (65,04% Furto / 34,96% Roubo), dividida em Treino (12.600 / 70%), Validação (2.700 / 15%) e Teste (2.700 / 15%).
- **Atributos de Entrada (53 features pós-OneHotEncoding e StandardScaler):** BAIRRO_AGRUPADO, PERIODO_DIA, DIA_SEMANA, TIPOLOCAL_AGRUPADO, HORA_NUM e FIM_DE_SEMANA. Variáveis que revelavam diretamente o crime (RUBRICA, DESC_LEI, DESDOBRAMENTO, CIRCUNSTANCIA) foram excluídas das entradas para evitar data leakage.

## 3. Comparação de Arquiteturas (Conjunto de Validação)
| Arquitetura | Camadas Ocultas | Regularização / Ajustes | Acurácia | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Arquitetura 2 (Vencedora)** | 3 (128 - 64 - 32, ReLU) | EarlyStopping + ClassWeight | **0.6904** | 0.5493 | 0.6377 | **0.5902** |
| **Arquitetura 1 (Baseline)** | 1 (32, ReLU) | EarlyStopping + ClassWeight | 0.6841 | 0.5402 | **0.6472** | 0.5889 |
| **Arquitetura 3 (Regularizada)** | 2 (64 - 32, ReLU) | Dropout (0.3/0.2) + L2 + BatchNorm | 0.6781 | **0.6904** | 0.1441 | 0.2384 |

- **Desempenho Final da Arquitetura 2 no Conjunto de Teste (2.700 amostras):** Acurácia = 69,81% | Precision = 56,04% | Recall = 63,35% | F1-Score = 0,5947.

## 4. Como Executar Localmente
1. Instale as dependências: `pip install -r requirements.txt`
2. Inicie a aplicação web: `streamlit run app.py`
