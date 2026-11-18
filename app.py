import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

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
def carregar_modelo_e_preprocessador():
    modelo = keras.models.load_model("modelo_mlp_sp.keras")
    
    # Categorias exatas extraídas do artefatos_web.pkl (ordem alfabética exata das 51 colunas One-Hot do treino)
    bairros = [
        'BARRA FUNDA', 'BELA VISTA', 'BOM RETIRO', 'BRÁS', 'CAPAO REDONDO',
        'CENTRO', 'CENTRO HISTÓRICO DE SÃO PAULO', 'CONSOLACAO', 'CONSOLAÇÃO',
        'IPIRANGA', 'ITAIM BIBI', 'ITAQUERA', 'JARDIM PAULISTA', 'LAPA',
        'LIBERDADE', 'MOOCA', 'OUTROS BAIRROS', 'PINHEIROS', 'REPUBLICA',
        'REPÚBLICA', 'SANTA CECILIA', 'SANTANA', 'SANTO AMARO', 'TATUAPÉ',
        'VILA GUILHERME', 'VILA MARIANA'
    ]
    periodos = ['A NOITE', 'A TARDE', 'DE MADRUGADA', 'EM HORA INCERTA', 'PELA MANHÃ']
    dias_treino = ['DOMINGO', 'QUARTA-FEIRA', 'QUINTA-FEIRA', 'SABADO', 'SEGUNDA-FEIRA', 'SEXTA-FEIRA', 'TERCA-FEIRA']
    dias_exibicao = ['Segunda-feira', 'Terca-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sabado', 'Domingo']
    tipos_local = [
        'CENTRO COMERCIAL/EMPRESARIAL', 'COMÉRCIO E SERVIÇOS', 'ESTACIONAMENTO/GARAGEM',
        'LAZER E RECREAÇÃO', 'OUTROS', 'OUTROS LOCAIS', 'RESIDÊNCIA',
        'RESTAURANTE E AFINS', 'RODOVIA/ESTRADA', 'SAÚDE', 'SHOPPING CENTER',
        'TERMINAL/ESTAÇÃO', 'VIA PÚBLICA'
    ]
    
    features_categoricas = ['BAIRRO_AGRUPADO', 'PERIODO_DIA', 'DIA_SEMANA', 'TIPOLOCAL_AGRUPADO']
    features_numericas = ['HORA_NUM', 'FIM_DE_SEMANA']
    
    # Reconstrói o OneHotEncoder exatamente com as 51 categorias do treino
    ohe = OneHotEncoder(
        categories=[bairros, periodos, dias_treino, tipos_local],
        handle_unknown='ignore',
        sparse_output=False
    )
    scaler = StandardScaler()
    
    preprocessador = ColumnTransformer(transformers=[
        ('cat', ohe, features_categoricas),
        ('num', scaler, features_numericas)
    ])
    
    # Inicializa a estrutura do ColumnTransformer
    df_init = pd.DataFrame([{
        'BAIRRO_AGRUPADO': bairros[0],
        'PERIODO_DIA': periodos[0],
        'DIA_SEMANA': dias_treino[0],
        'TIPOLOCAL_AGRUPADO': tipos_local[0],
        'HORA_NUM': 14.0,
        'FIM_DE_SEMANA': 0
    }])
    preprocessador.fit(df_init)
    
    # Injeta os parâmetros estatísticos exatos do conjunto de treino (12.600 amostras) extraídos do artefatos_web.pkl
    scaler_ajustado = preprocessador.named_transformers_['num']
    scaler_ajustado.mean_ = np.array([13.48936508, 0.28738095])
    scaler_ajustado.var_ = np.array([37.79822937, 0.20479317])
    scaler_ajustado.scale_ = np.array([6.14802646, 0.45254079])
    scaler_ajustado.n_samples_seen_ = 12600
    
    return modelo, preprocessador, bairros, periodos, dias_exibicao, tipos_local

modelo, preprocessador, bairros, periodos, dias_semana, tipos_local = carregar_modelo_e_preprocessador()

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
        bairro = st.selectbox("Bairro da Ocorrência (SP Capital)", options=bairros)
        periodo = st.selectbox("Período do Dia", options=periodos)
        dia_semana = st.selectbox("Dia da Semana", options=dias_semana)
        
    with col2:
        tipo_local = st.selectbox("Tipo de Local", options=tipos_local)
        hora_num = st.slider("Hora Aproximada da Ocorrência (0h às 23h)", min_value=0, max_value=23, value=19)
    
    submetido = st.form_submit_button("🔍 Classificar Ocorrência", use_container_width=True)

# Resultado da pesquisa aparece logo abaixo do formulário (e acima da nota do rodapé)
if submetido:
    fim_de_semana = 1 if str(dia_semana).upper() in ["SABADO", "DOMINGO", "SÁBADO"] else 0
    
    df_input = pd.DataFrame([{
        "BAIRRO_AGRUPADO": bairro,
        "PERIODO_DIA": periodo,
        "DIA_SEMANA": str(dia_semana).upper(),
        "TIPOLOCAL_AGRUPADO": tipo_local,
        "HORA_NUM": float(hora_num),
        "FIM_DE_SEMANA": int(fim_de_semana)
    }])
    
    # Transforma gerando exatamente as 53 features alinhadas com a rede MLP
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
