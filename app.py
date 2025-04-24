import streamlit as st
import pandas as pd
import pickle
import joblib

# Carregar modelo e scaler
with open('modelo_final.pkl', 'rb') as f:
    modelo_dict = pickle.load(f)

modelo = modelo_dict['modelo']
colunas_modelo = modelo_dict['colunas']

scaler = joblib.load('scaler.pkl')

st.title("✈️ Previsão de Atraso na Decolagem")

st.write("Insira os dados do voo abaixo:")

# Entradas do usuário
empresa = st.text_input("Sigla ICAO da Empresa Aérea")
numero_voo = st.number_input("Número do voo", step=1)
modelo_equip = st.text_input("Modelo da Aeronave")
origem = st.text_input("Sigla ICAO da Origem")
destino = st.text_input("Sigla ICAO do Destino")

if st.button("Prever"):
    dados = pd.DataFrame([{
        'sigla_icao_empresa_aerea': empresa.upper(),
        'numero_voo': numero_voo,
        'modelo_equipamento': modelo_equip.upper(),
        'sigla_icao_origem': origem.upper(),
        'sigla_icao_destino': destino.upper()
    }])

    # One-hot encoding + alinhamento
    dados_encoded = pd.get_dummies(dados)
    for col in colunas_modelo:
        if col not in dados_encoded.columns:
            dados_encoded[col] = 0
    dados_encoded = dados_encoded[colunas_modelo]

    # Escalonar
    dados_scaled = scaler.transform(dados_encoded)

    # Previsão
    prob = modelo.predict_proba(dados_scaled)[0][1]
    st.metric("📊 Chance de Atraso", f"{prob*100:.2f}%")

