# Segurança Pública Inteligente em São Paulo: Classificação de Ocorrências com MLP

Projeto desenvolvido para a **Global Solution** da disciplina de *Aprendizado Profundo com Redes Neurais Convolucionais* (FIAP).

## 🌐 Acesso à Aplicação Web
Acesse a aplicação em funcionamento: [https://gs-mlp-seguranca-sp-qsit7kr2bcmqrtbddayvyi.streamlit.app/](https://gs-mlp-seguranca-sp-qsit7kr2bcmqrtbddayvyi.streamlit.app/)

---

## 1. Objetivo do Projeto
Desenvolver, treinar e comparar três arquiteturas de redes neurais **Multilayer Perceptron (MLP)** para classificar ocorrências policiais registradas na capital de São Paulo entre **Roubo (1)** e **Furto (0)**. 

⚠️ *Nota metodológica:* O modelo analisa o perfil e as características contextuais de uma ocorrência (registrada ou hipotética) para estimar a sua compatibilidade com roubo ou furto, e não o risco futuro de um crime acontecer.

## 2. Fonte de Dados e Tratamento
* **Fonte Pública:** Secretaria da Segurança Pública do Estado de São Paulo (SSP-SP) – Portal de Estatísticas[cite: 16].
* **Base Utilizada:** `CelularesSubtraidos_2026.xlsx` (Aba `CELULAR_2026`).
* **Recorte e Amostragem:** 
  * Filtragem restrita ao município de São Paulo (Capital), resultando em 151.170 registros.
  * Tratamento de duplicidades por Boletim de Ocorrência, totalizando 130.851 BOs únicos.
  * Amostra balanceada de **18.000 observações** (65,04% Furto / 34,96% Roubo), dividida em Treino (70% / 12.600), Validação (15% / 2.700) e Teste (15% / 2.700).
* **Atributos de Entrada ($X$):** `BAIRRO_AGRUPADO`, `PERIODO_DIA`, `DIA_SEMANA`, `TIPOLOCAL_AGRUPADO`, `HORA_NUM` e `FIM_DE_SEMANA` (totalizando 53 features após *OneHotEncoding* e *StandardScaler*). Variáveis que revelavam a natureza da infração foram estritamente excluídas para evitar *data leakage*[cite: 17].

## 3. Comparação de Arquiteturas (Conjunto de Validação)

| Arquitetura | Camadas Ocultas | Ajustes e Regularização | Acurácia | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Arquitetura 2 (Vencedora)** | 3 (128 - 64 - 32, ReLU) | EarlyStopping + ClassWeight | **0.6904** | 0.5493 | 0.6377 | **0.5902** |
| **Arquitetura 1 (Baseline)** | 1 (32, ReLU) | EarlyStopping + ClassWeight | 0.6841 | 0.5402 | 0.6472 | 0.5889 |
| **Arquitetura 3 (Regularizada)** | 2 (64 - 32, ReLU) | Dropout (0.3/0.2) + L2 + BatchNorm | 0.6781 | 0.6904 | 0.1441 | 0.2384 |

* **Desempenho Final no Conjunto de Teste (2.700 amostras):** Acurácia = **69,81%** | Precision = **56,04%** | Recall = **63,35%** | F1-Score = **0,5947**.

## 4. Como Executar Localmente
Para rodar o projeto na sua máquina:

1. Clone o repositório e instale as dependências:
   ```bash
   pip install -r requirements.txt
