import streamlit as st
import pandas as pd
import pickle
import joblib

st.set_page_config(page_title="Previsão de Atraso", layout="wide")

# Carregar modelo e scaler
with open('modelo_final.pkl', 'rb') as f:
    modelo_dict = pickle.load(f)

modelo = modelo_dict['modelo']
colunas_modelo = modelo_dict['colunas']
scaler = joblib.load('scaler.pkl')

# Carregar dados
df = pd.read_csv("dados_tratados.csv")
if 'atraso_decolagem' in df.columns:
    df['atraso'] = df['atraso_decolagem']
else:
    st.error("❌ A coluna 'atraso_decolagem' não foi encontrada na base de dados.")
    st.stop()

# Menu lateral
menu = st.sidebar.radio("Menu", ["Principal", "Início", "Estatísticas", "Previsão", "Sobre o Modelo"])

# --------------------- TUDO NA MESMA PÁGINA ---------------------
if menu == "Principal":

    st.video("Aviao.mp4", start_time=0, format="mp4", loop=True)
    st.title("✈️ Previsão de Atraso na Decolagem")
    st.markdown("""
                Bem-vindo ao nosso sistema de previsão de atrasos em voos!
                Este site utiliza **dados reais da ANAC** e um modelo de aprendizado de máquina treinado para estimar a **chance de um voo atrasar na decolagem**, com base em características como empresa aérea, número do voo e aeroporto de origem.
                """)


    with st.expander("📊 Estatísticas de Atrasos", expanded=False):
        col_esq, col_dir = st.columns(2)

        with col_esq:
            dados_empresas = df.groupby('sigla_icao_empresa_aerea')['atraso'].mean().sort_values(ascending=False).head(5) * 100
            st.subheader("Empresas com Maior Percentual de Atrasos")
            st.bar_chart(dados_empresas)

        with col_dir:
            dados_destinos = df.groupby('sigla_icao_destino')['atraso'].mean().sort_values(ascending=False).head(5) * 100
            st.subheader("Destinos com Maior Percentual de Atrasos")
            st.bar_chart(dados_destinos)

    st.markdown("---")

    st.subheader("🔎 Faça sua Previsão de Atraso")

    col1, col2 = st.columns([2, 1])
    with col1:
        empresas_disponiveis = sorted(df['sigla_icao_empresa_aerea'].dropna().unique())
        aeroportos_origem = sorted(df['sigla_icao_origem'].dropna().unique())

        empresa = st.selectbox("Sigla da Empresa Aérea", options=empresas_disponiveis, index=None, placeholder="Digite ou selecione")
        numero_voo = st.text_input("Número do Voo")
        origem = st.selectbox("Aeroporto de Origem", options=aeroportos_origem, index=None, placeholder="Digite ou selecione")

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

                dados_encoded = pd.get_dummies(dados_input)
                for col in colunas_modelo:
                    if col not in dados_encoded.columns:
                        dados_encoded[col] = 0
                dados_encoded = dados_encoded[colunas_modelo]
                dados_scaled = scaler.transform(dados_encoded)
                prob = modelo.predict_proba(dados_scaled)[0][1]
                st.metric("Chance de Atraso", f"{prob * 100:.2f}%")
            else:
                st.warning("Preencha todos os campos.")

    with col2:
        st.image("imagem.jpg", caption="Imagem ilustrativa", use_container_width=True)

    st.markdown("---")

    with st.expander("🧠 Sobre o Modelo"):
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

# --------------------- MODO MENU SEPARADO ---------------------
elif menu == "Início":
    st.video("Aviao.mp4", start_time=0, format="mp4", loop=True)
    st.title("✈️ Previsão de Atraso na Decolagem")

elif menu == "Estatísticas":
    st.title("📊 Estatísticas de Atrasos")
    col_esq, col_dir = st.columns(2)

    with col_esq:
        dados_empresas = df.groupby('sigla_icao_empresa_aerea')['atraso'].mean().sort_values(ascending=False).head(5) * 100
        st.subheader("Empresas com Maior Percentual de Atrasos")
        st.bar_chart(dados_empresas)

    with col_dir:
        dados_destinos = df.groupby('sigla_icao_destino')['atraso'].mean().sort_values(ascending=False).head(5) * 100
        st.subheader("Destinos com Maior Percentual de Atrasos")
        st.bar_chart(dados_destinos)

elif menu == "Previsão":
    st.title("🔎 Previsão de Atraso")

    col1, col2 = st.columns([2, 1])
    with col1:
        empresas_disponiveis = sorted(df['sigla_icao_empresa_aerea'].dropna().unique())
        aeroportos_origem = sorted(df['sigla_icao_origem'].dropna().unique())

        empresa = st.selectbox("Sigla da Empresa Aérea", options=empresas_disponiveis, index=None, placeholder="Digite ou selecione")
        numero_voo = st.text_input("Número do Voo")
        origem = st.selectbox("Aeroporto de Origem", options=aeroportos_origem, index=None, placeholder="Digite ou selecione")

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

                dados_encoded = pd.get_dummies(dados_input)
                for col in colunas_modelo:
                    if col not in dados_encoded.columns:
                        dados_encoded[col] = 0
                dados_encoded = dados_encoded[colunas_modelo]
                dados_scaled = scaler.transform(dados_encoded)
                prob = modelo.predict_proba(dados_scaled)[0][1]
                st.metric("Chance de Atraso", f"{prob * 100:.2f}%")
            else:
                st.warning("Preencha todos os campos.")

    with col2:
        st.image("imagem.jpg", caption="Imagem ilustrativa", use_container_width=True)

elif menu == "Sobre o Modelo":
    st.title("🧠 Sobre o Modelo de Previsão")
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

     Nosso objetivo é ajudar no planejamento de operações aéreas e melhorar a experiência dos passageiros!
    """)

O modelo prevê a **chance (%) de um voo sofrer atraso na decolagem**.

 Nosso objetivo é ajudar no planejamento de operações aéreas e melhorar a experiência dos passageiros!
""")
