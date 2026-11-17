import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Classificador SSP-SP | MLP", page_icon="🚓", layout="centered")

@st.cache_resource
def carregar_modelo_e_dados():
    modelo = keras.models.load_model('modelo_mlp_sp.keras')
    
    bairros = [
        'OUTROS BAIRROS', 'SANTA CECILIA', 'LIBERDADE', 'BELA VISTA', 'CONSOLACAO', 
        'REPUBLICA', 'PERDIZES', 'PINHEIROS', 'MOEMA', 'ITAIM BIBI', 'VILA MARIANA', 
        'SAO PAULO', 'BARRA FUNDA', 'CENTRO', 'JARDIM PAULISTA', 'MOOCA', 
        'IPIRANGA', 'SANTANA', 'TUCURUVI', 'LAPA', 'BUTANTA', 'CAMPO BELO', 'ITAQUERA'
    ]
    periodos = ['A NOITE', 'A TARDE', 'DE MADRUGADA', 'EM HORA INCERTA', 'PELA MANHÃ']
    dias_semana = ['Segunda-feira', 'Terca-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sabado', 'Domingo']
    tipos_local = [
        'Via Pública', 'Residência', 'Comércio', 'Estabelecimento Comercial', 
        'Outros', 'Terminal de Ônibus', 'Estação de Metrô', 'Rodovia', 'Estacionamento'
    ]
    
    features_categoricas = ['BAIRRO_AGRUPADO', 'PERIODO_DIA', 'DIA_SEMANA', 'TIPOLOCAL_AGRUPADO']
    features_numericas = ['HORA_NUM', 'FIM_DE_SEMANA']
    
    transformador_cat = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    transformador_num = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessador = ColumnTransformer(transformers=[
        ('cat', transformador_cat, features_categoricas),
        ('num', transformador_num, features_numericas)
    ])
    
    df_dummy = pd.DataFrame({
        'BAIRRO_AGRUPADO': bairros,
        'PERIODO_DIA': [periodos[0]] * len(bairros),
        'DIA_SEMANA': [dias_semana[0]] * len(bairros),
        'TIPOLOCAL_AGRUPADO': [tipos_local[0]] * len(bairros),
        'HORA_NUM': [12.0] * len(bairros),
        'FIM_DE_SEMANA': [0] * len(bairros)
    })
    
    preprocessador.fit(df_dummy)
    
    return modelo, preprocessador, bairros, periodos, dias_semana, tipos_local

modelo, preprocessador, bairros, periodos, dias_semana, tipos_local = carregar_modelo_e_dados()

st.title("🚓 Classificação Inteligente de Ocorrências em São Paulo (MLP)")
st.markdown(
    "Esta aplicação utiliza uma rede neural **Multilayer Perceptron (MLP)** treinada com dados públicos da **SSP-SP (Celulares Subtraídos - 2026)** "
    "para classificar se as características de uma ocorrência registada na capital paulista são mais compatíveis com **Furto** ou **Roubo**."
)
st.info("⚠️ **Nota metodológica:** O modelo analisa o perfil de uma ocorrência e estima a compatibilidade com a natureza criminal (roubo vs. furto).")

with st.form("form_ocorrencia"):
    st.subheader("Informe as características da ocorrência")
    col1, col2 = st.columns(2)
    
    with col1:
        bairro = st.selectbox("Bairro da Ocorrência (SP Capital)", options=bairros)
        periodo = st.selectbox("Período do Dia", options=periodos)
        dia_semana = st.selectbox("Dia da Semana", options=dias_semana)
        
    with col2:
        tipo_local = st.selectbox("Tipo de Local", options=tipos_local)
        hora_num = st.slider("Hora Aproximada da Ocorrência (0h às 23h)", min_value=0, max_value=23, value=19)
    
    submetido = st.form_submit_button("🔍 Classificar Ocorrência", use_container_width=True)

if submetido:
    fim_de_semana = 1 if dia_semana in ['SABADO', 'DOMINGO', 'Sabado', 'Domingo'] else 0
    
    df_input = pd.DataFrame([{
        'BAIRRO_AGRUPADO': bairro,
        'PERIODO_DIA': periodo,
        'DIA_SEMANA': dia_semana,
        'TIPOLOCAL_AGRUPADO': tipo_local,
        'HORA_NUM': float(hora_num),
        'FIM_DE_SEMANA': int(fim_de_semana)
    }])
    
    X_input_proc = preprocessador.transform(df_input)
    
    # Alinhamento automático de dimensões com o modelo treinado
    expected_features = modelo.input_shape[1]
    if hasattr(X_input_proc, "toarray"):
        X_input_proc = X_input_proc.toarray()
        
    if X_input_proc.shape[1] < expected_features:
        X_padded = np.zeros((X_input_proc.shape[0], expected_features))
        X_padded[:, :X_input_proc.shape[1]] = X_input_proc
        X_input_proc = X_padded
    elif X_input_proc.shape[1] > expected_features:
        X_input_proc = X_input_proc[:, :expected_features]

    prob_roubo = float(modelo.predict(X_input_proc, verbose=0)[0][0])
    prob_furto = 1.0 - prob_roubo
    
    classe_estimada = "ROUBO (Art. 157)" if prob_roubo >= 0.5 else "FURTO (Art. 155)"
    
    st.divider()
    st.subheader(f"Resultado Estimado pela MLP: **{classe_estimada}**")
    
    c1, c2 = st.columns(2)
    c1.metric("Probabilidade atribuída a ROUBO", f"{prob_roubo * 100:.1f}%")
    c2.metric("Probabilidade atribuída a FURTO", f"{prob_furto * 100:.1f}%")
    
    st.progress(prob_roubo, text=f"Escala de Compatibilidade: Furto ({prob_furto*100:.1f}%) ➔ Roubo ({prob_roubo*100:.1f}%)")
