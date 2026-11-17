import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras

st.set_page_config(page_title="Classificador SSP-SP | MLP", page_icon="🚓", layout="centered")

@st.cache_resource
def carregar_modelo_e_artefatos():
    modelo = keras.models.load_model("modelo_mlp_sp.keras")
    artefatos = joblib.load("artefatos_web.pkl")
    return modelo, artefatos

modelo, artefatos = carregar_modelo_e_artefatos()
preprocessador = artefatos["preprocessador"]

st.title("🚓 Classificação Inteligente de Ocorrências em São Paulo (MLP)")
st.markdown("Esta aplicação utiliza uma rede neural **Multilayer Perceptron (MLP)** treinada com dados públicos da **SSP-SP (Celulares Subtraídos - 2026)** para classificar se as características de uma ocorrência registrada na capital paulista são mais compatíveis com **Furto** ou **Roubo**.")
st.info("⚠️ **Nota metodológica:** O modelo analisa o perfil de uma ocorrência que já aconteceu (ou hipotética) e estima a compatibilidade com a natureza criminal (roubo vs. furto). Ele não prevê o risco de um crime futuro ocorrer.")

with st.form("form_ocorrencia"):
    st.subheader("Informe as características da ocorrência")
    col1, col2 = st.columns(2)
    with col1:
        bairro = st.selectbox("Bairro da Ocorrência (SP Capital)", options=artefatos["bairros"])
        periodo = st.selectbox("Período do Dia", options=artefatos["periodos"])
        dia_semana = st.selectbox("Dia da Semana", options=artefatos["dias_semana"])
    with col2:
        tipo_local = st.selectbox("Tipo de Local", options=artefatos["tipos_local"])
        hora_num = st.slider("Hora Aproximada da Ocorrência (0h às 23h)", min_value=0, max_value=23, value=19)
    submetido = st.form_submit_button("🔍 Classificar Ocorrência", use_container_width=True)

if submetido:
    fim_de_semana = 1 if dia_semana in ["SABADO", "DOMINGO", "Sabado", "Domingo"] else 0
    df_input = pd.DataFrame([{
        "BAIRRO_AGRUPADO": bairro,
        "PERIODO_DIA": periodo,
        "DIA_SEMANA": dia_semana,
        "TIPOLOCAL_AGRUPADO": tipo_local,
        "HORA_NUM": hora_num,
        "FIM_DE_SEMANA": fim_de_semana
    }])
    X_input_proc = preprocessador.transform(df_input)
    prob_roubo = float(modelo.predict(X_input_proc, verbose=0)[0][0])
    prob_furto = 1.0 - prob_roubo
    classe_estimada = "ROUBO (Art. 157)" if prob_roubo >= 0.5 else "FURTO (Art. 155)"
    st.divider()
    st.subheader(f"Resultado Estimado pela MLP: **{classe_estimada}**")
    c1, c2 = st.columns(2)
    c1.metric("Probabilidade atribuída a ROUBO", f"{prob_roubo * 100:.1f}%")
    c2.metric("Probabilidade atribuída a FURTO", f"{prob_furto * 100:.1f}%")
    st.progress(prob_roubo, text=f"Compatibilidade: Furto ({prob_furto*100:.1f}%) ➔ Roubo ({prob_roubo*100:.1f}%)")
    st.caption(f"**Interpretação correta:** Considerando que uma ocorrência aconteceu em **{bairro}** ({tipo_local}), na **{dia_semana}** ({periodo}, ~{hora_num}h), a rede MLP atribui **{prob_roubo*100:.1f}%** de probabilidade à classe Roubo e **{prob_furto*100:.1f}%** à classe Furto.")