import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

valores_possiveis = {
    'informacoes_pessoais_sexo': ['Masculino', 'Feminino'], 
    'informacoes_pessoais_estado_civil': ['Solteiro', 'Casado', 'União Estável', 'Divorciado', 'Separado Judicialmente', 'Viúvo'], 
    'informacoes_pessoais_pcd': ['Não', 'Sim'], 
    'informacoes_pessoais_endereco': ['rio de janeiro', 'são paulo', 'bahia', 'minas gerais', 'distrito federal', 'santa catarina', 'paraná', 'rio grande do sul', 'maranhão', 'pernambuco', 'ceará', 'pará', 'paraíba', 'piauí', 'goiás', 'alagoas', 'rondônia', 'rio grande do norte', 'sergipe', 'roraima', 'mato grosso do sul', 'amazonas', 'tocantins', 'mato grosso', 'amapá', 'acre'], 'formacao_e_idiomas_nivel_academico': ['Ensino Superior Completo', 'Pós Graduação Completo', 'Ensino Médio Completo', 'Mestrado Completo', 'Ensino Superior Cursando', 'Pós Graduação Incompleto', 'Ensino Superior Incompleto', 'Ensino Técnico Completo', 'Pós Graduação Cursando', 'Ensino Técnico Cursando', 'Mestrado Incompleto', 'Ensino Fundamental Completo', 'Ensino Médio Incompleto', 'Mestrado Cursando', 'Doutorado Incompleto', 'Doutorado Completo', 'Doutorado Cursando', 'Ensino Médio Cursando', 'Ensino Técnico Incompleto', 'Ensino Fundamental Incompleto', 'Ensino Fundamental Cursando'], 'formacao_e_idiomas_nivel_ingles': ['Intermediário', 'Avançado', 'Fluente', 'Básico', 'Nenhum'], 'formacao_e_idiomas_nivel_espanhol': ['Básico', 'Intermediário', 'Nenhum', 'Fluente', 'Avançado']
}

modelo = joblib.load("modelo_ia_decision.joblib")

st.title("Cadastro de Candidato")

with st.form("formulario_candidato"):
    st.subheader("Preencha seus dados")
    sexo = st.selectbox("Sexo", valores_possiveis['informacoes_pessoais_sexo'])
    estado_civil = st.selectbox("Estado Civil", valores_possiveis['informacoes_pessoais_estado_civil'])
    pcd = st.selectbox("Possui deficiência (PCD)?", valores_possiveis['informacoes_pessoais_pcd'])
    endereco = st.selectbox("Estado", valores_possiveis['informacoes_pessoais_endereco'])
    nivel_academico = st.selectbox("Formação Acadêmica", valores_possiveis['formacao_e_idiomas_nivel_academico'])
    nivel_ingles = st.selectbox("Nível de Inglês", valores_possiveis['formacao_e_idiomas_nivel_ingles'])
    nivel_espanhol = st.selectbox("Nível de Espanhol", valores_possiveis['formacao_e_idiomas_nivel_espanhol'])

    enviar = st.form_submit_button("Enviar")

if enviar:
    candidato = pd.DataFrame([{
        "informacoes_pessoais_sexo": sexo,
        "informacoes_pessoais_estado_civil": estado_civil,
        "informacoes_pessoais_pcd": pcd,
        "informacoes_pessoais_endereco": endereco,
        "formacao_e_idiomas_nivel_academico": nivel_academico,
        "formacao_e_idiomas_nivel_ingles": nivel_ingles,
        "formacao_e_idiomas_nivel_espanhol": nivel_espanhol
    }])

    candidato["data_cadastro"] = datetime.now()

    try:
        base = pd.read_csv("base_candidatos.csv")
        base = pd.concat([base, candidato], ignore_index=True)
    except FileNotFoundError:
        base = candidato

    base.to_csv("base_candidatos.csv", index=False)

    prob = modelo.predict_proba(candidato)[0][1]
    st.success(f"Probabilidade estimada de contratação: {prob:.1%}")

    def gerar_radar(candidato, modelo, valores_possiveis):
        radar_scores = {}

        for coluna, opcoes in valores_possiveis.items():
            if coluna not in candidato:
                continue
            resultados = []
            for val in opcoes:
                temp = candidato.copy()
                temp[coluna] = val
                p = modelo.predict_proba(pd.DataFrame([temp]))[0][1]
                resultados.append((val, p))
            valor_real = candidato[coluna]
            prob_real = next((p for v, p in resultados if v == valor_real), None)
            max_prob = max(p for _, p in resultados)
            score = 10 * prob_real / max_prob if (max_prob and prob_real is not None) else 0
            radar_scores[coluna.replace('_', ' ').title()] = score

        labels = list(radar_scores.keys())
        values = list(radar_scores.values())
        values += values[:1]
        angles = [n / float(len(labels)) * 2 * np.pi for n in range(len(labels))]
        angles += angles[:1]

        fig = plt.figure(figsize=(6,6))
        ax = plt.subplot(111, polar=True)
        plt.xticks(angles[:-1], labels, color='grey', size=10)
        ax.plot(angles, values, linewidth=2)
        ax.fill(angles, values, 'skyblue', alpha=0.4)
        plt.title(f"Radar de Perfil\nChance: {prob:.1%}", size=14, y=1.1)
        st.pyplot(fig)

    gerar_radar(candidato.iloc[0], modelo, valores_possiveis)