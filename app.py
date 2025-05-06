import streamlit as st
st.set_page_config(page_title="Previsão de Atraso", layout="wide")
import pandas as pd
import pickle
import joblib

# Carregar modelo e scaler
with open('modelo_final.pkl', 'rb') as f:
    modelo_dict = pickle.load(f)

modelo = modelo_dict['modelo']
colunas_modelo = modelo_dict['colunas']
scaler = joblib.load('scaler.pkl')

# Carregar os dados tratados diretamente do CSV
df = pd.read_csv("dados_tratados.csv")

# ✅ Ajuste: criar coluna 'atraso' a partir de 'atraso_decolagem'
if 'atraso_decolagem' in df.columns:
    df['atraso'] = df['atraso_decolagem']
else:
    st.error("❌ A coluna 'atraso_decolagem' não foi encontrada na base de dados.")
    st.stop()

# Vídeo com looping
st.video("Aviao.mp4", start_time=0, format="mp4", loop=True)

# Título
st.title("✈️ Previsão de Atraso na Decolagem")

# Layout para gráficos
col_esq, col_dir = st.columns(2)

# Estatísticas de Atrasos por Empresa
with col_esq:
    st.header("📊 Estatísticas de Atrasos")
    dados_empresas = df.groupby('sigla_icao_empresa_aerea')['atraso'].mean().sort_values(ascending=False).head(5) * 100
    st.subheader("Empresas com Maior Percentual de Atrasos")
    st.bar_chart(dados_empresas)

# Estatísticas de Atrasos por Destino
with col_dir:
    dados_destinos = df.groupby('sigla_icao_destino')['atraso'].mean().sort_values(ascending=False).head(5) * 100
    st.subheader("Destinos com Maior Percentual de Atrasos")
    st.bar_chart(dados_destinos)

st.divider()

# Layout de previsão
col1, col2 = st.columns([2, 1])  

with col1:
    st.header("🔎 Previsão de Atraso")

    # Inputs do usuário
    empresa = st.text_input("Sigla da Empresa Aérea", key='empresa')
    numero_voo = st.text_input("Número do Voo", key='num_voo')
    origem = st.text_input("Aeroporto de Origem", key='origem')

    modelo_equip = "A320"  
    destino = "SBSP"       

    if st.button("Prever"):
        if empresa and numero_voo and origem:
            dados_input = pd.DataFrame([{
                'sigla_icao_empresa_aerea': empresa.upper(),
                'numero_voo': int(numero_voo),
                'modelo_equipamento': modelo_equip.upper(),
                'sigla_icao_origem': origem.upper(),
                'sigla_icao_destino': destino.upper()
            }])

            # Codificação e alinhamento
            dados_encoded = pd.get_dummies(dados_input)
            for col in colunas_modelo:
                if col not in dados_encoded.columns:
                    dados_encoded[col] = 0
            dados_encoded = dados_encoded[colunas_modelo]

            # Escalonamento
            dados_scaled = scaler.transform(dados_encoded)

            # Previsão
            prob = modelo.predict_proba(dados_scaled)[0][1]
            st.metric("Chance de Atraso", f"{prob * 100:.2f}%")
        else:
            st.warning("Por favor, preencha todos os campos para realizar a previsão.")

with col2:
    st.image("imagem.jpg", caption="Imagem ilustrativa", use_container_width=True)

st.divider()

# Sobre o modelo
st.subheader("🧠 Sobre o Modelo de Previsão")
st.markdown("""
O modelo foi desenvolvido a partir de **dados abertos da ANAC**, utilizando informações históricas de voos para aprender padrões de atraso.

**Dados utilizados:**
- Empresa aérea
- Número do voo
- Origem e destino
- Modelo da aeronave

**Técnicas aplicadas:**
- Escalonamento e codificação de variáveis
- Algoritmos de classificação com output probabilístico

O modelo prevê a **chance (%) de um voo sofrer atraso na decolagem**.

⚡ Nosso objetivo é ajudar no planejamento de operações aéreas e melhorar a experiência dos passageiros!
""")
