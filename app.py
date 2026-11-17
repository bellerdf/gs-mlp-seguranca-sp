import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras

st.set_page_config(page_title="Classificador SSP-SP | MLP", page_icon="🚓", layout="centered")

# Ajuste visual para deixar o título em uma única linha e o layout mais fluido
st.markdown("""
    <style>
    .block-container {
        max-width: 860px;
        padding-top: 2.5rem;
    }
    .titulo-linha-unica {
        font-size: 1.75rem;
        font-weight: 700;
        white-space: nowrap;
        margin-bottom: 0.4rem;
    }
    .subtitulo-fluido {
        font-size: 0.98rem;
        color: #4a4a4a;
        margin-bottom: 1.8rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def carregar_modelo_e_artefatos():
    modelo = keras.models.load_model("modelo_mlp_sp.keras")
    artefatos = joblib.load("artefatos_web.pkl")
    return modelo, artefatos

modelo, artefatos = carregar_modelo_e_artefatos()
preprocessador = artefatos["preprocessador"]

# Cabeçalho em linha única e texto logo abaixo seguindo o alinhamento
st.markdown('<div class="titulo-linha-unica">🚓 Classificação Inteligente de Ocorrências em São Paulo (MLP)</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitulo-fluido">Esta aplicação utiliza uma rede neural <b>Multilayer Perceptron (MLP)</b> treinada com dados públicos da '
    '<b>SSP-SP (Celulares Subtraídos - 2026)</b> para classificar se as características de uma ocorrência registrada na capital paulista são mais compatíveis com <b>Furto</b> ou <b>Roubo</b>.</div>',
    unsafe_allow_html=True
)

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

# Resultado da pesquisa aparece logo abaixo do formulário (e acima da nota do rodapé)
if submetido:
    # Normaliza a string para maiúsculas para bater com o formato salvo no treino ("SABADO", "DOMINGO")
    fim_de_semana = 1 if str(dia_semana).upper() in ["SABADO", "DOMINGO", "SÁBADO"] else 0
    
    df_input = pd.DataFrame([{
        "BAIRRO_AGRUPADO": bairro,
        "PERIODO_DIA": periodo,
        "DIA_SEMANA": str(dia_semana).upper(),
        "TIPOLOCAL_AGRUPADO": tipo_local,
        "HORA_NUM": float(hora_num),
        "FIM_DE_SEMANA": int(fim_de_semana)
    }])
    
    # Transforma os dados usando exatamente o pipeline ajustado no conjunto de treino (53 features exatas)
    X_input_proc = preprocessador.transform(df_input)
    
    prob_roubo = float(modelo.predict(X_input_proc, verbose=0)[0][0])
    prob_furto = 1.0 - prob_roubo
    
    classe_estimada = "ROUBO (Art. 157)" if prob_roubo >= 0.5 else "FURTO (Art. 155)"
    
    st.divider()
    st.subheader(f"Resultado Estimado pela MLP: **{classe_estimada}**")
    
    c1, c2 = st.columns(2)
    c1.metric("Probabilidade atribuída a ROUBO", f"{prob_roubo * 100:.1f}%")
    c2.metric("Probabilidade atribuída a FURTO", f"{prob_furto * 100:.1f}%")
    
    st.progress(prob_roubo, text=f"Escala de Compatibilidade: Furto ({prob_furto*100:.1f}%) ➔ Roubo ({prob_roubo*100:.1f}%)")
    st.caption(
        f"**Interpretação correta:** Considerando que uma ocorrência aconteceu em **{bairro}** ({tipo_local}), "
        f"na **{dia_semana}** ({periodo}, ~{hora_num}h), a rede MLP atribui **{prob_roubo*100:.1f}%** de probabilidade "
        f"à classe Roubo e **{prob_furto*100:.1f}%** à classe Furto."
    )

# Rodapé fixo no final da página (sempre abaixo dos resultados)
st.write("")
st.info("⚠️ **Nota metodológica:** O modelo analisa o perfil de uma ocorrência que já aconteceu (ou hipotética) e estima a compatibilidade com a natureza criminal (roubo vs. furto). Ele não prevê o risco de um crime futuro ocorrer.")
