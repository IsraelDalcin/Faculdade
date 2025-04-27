#!pip install streamlit openai
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from openai import OpenAI
from pycaret.classification import *  # load_model, predict_model

# Configurações iniciais do Streamlit
st.set_page_config(page_title='Simulador - Case Ifood',
                   page_icon='./images/logo_fiap.png',
                   layout='wide',
                   initial_sidebar_state='expanded')

st.title('Simulador - Conversão de Vendas')

# Explicação
with st.expander('Descrição do App', expanded=False):
    st.write('O objetivo principal deste app é simular as conversões de vendas com ajuste dinâmico de threshold.')

# Sidebar
st.sidebar.write('Configurações')
with st.sidebar:
    c1, c2 = st.columns([.3, .7])
    c1.image('./images/logo_fiap.png', width=100)
    c2.subheader('Auto ML - Fiap [v1]')
    database = st.radio('Fonte dos dados de entrada (X):', ('CSV', 'Online'), horizontal=True)

    if database == 'CSV':
        st.info('Upload do CSV')
        file = st.file_uploader('Selecione o arquivo CSV', type='csv')


# Função para interpretar o threshold via IA
def interpretar_threshold(comando_usuario):
    prompt = f"Extraia apenas o valor numérico do threshold (entre 0 e 1) baseado no comando: '{comando_usuario}'."
    try:
        client = OpenAI(api_key="")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "Você é um assistente que extrai valores numéricos de thresholds entre 0 e 1."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        resposta = response.choices[0].message.content.strip()
        valor = float(resposta.replace(",", "."))
        if 0 <= valor <= 1:
            return ["Novo threshold interpretado", valor, True]
        else:
            return ["Por favor, informe um valor entre 0 e 1. Utilizando valor padrão", 0.5, False]
    except Exception as e:
        #st.error(f"Erro ao interpretar comando: {e}")
        return ["Por favor, informe um valor entre 0 e 1. Utilizando valor padrão", 0.5, False]


# Tela principal
if database == 'CSV':
    if file:
        # Carregamento do CSV
        Xtest = pd.read_csv(file)

        # Carregamento / instanciamento do modelo pkl
        mdl_rf = load_model('./pickle/pickle_rf_pycaret2')

        # Predict do modelo
        ypred = predict_model(mdl_rf, data=Xtest, raw_score=True)

        with st.expander('Visualizar CSV carregado:', expanded=False):
            c1, _ = st.columns([2, 4])
            qtd_linhas = c1.slider('Visualizar quantas linhas do CSV:',
                                   min_value=5,
                                   max_value=Xtest.shape[0],
                                   step=10,
                                   value=5)
            st.dataframe(Xtest.head(qtd_linhas))

        with st.expander('Visualizar Predições:', expanded=True):
            # Campo de texto para comando
            comando_usuario = st.text_input('Digite o comando para alterar o threshold (ex: "Aumente para 0.7"):', '')

            # Threshold inicial
            treshold = 0.5

            if comando_usuario:
                msg, treshold, success = interpretar_threshold(comando_usuario)

                if success:
                    st.success(f"{msg}: {treshold}")
                else:
                    st.warning(f"{msg}: {treshold}")


            # Slider para ajuste fino, já usando o threshold interpretado
            treshold = st.slider('Ajuste manual do Threshold:',
                                 min_value=0.0,
                                 max_value=1.0,
                                 step=0.01,
                                 value=treshold)

            qtd_true = ypred.loc[ypred['prediction_score_1'] > treshold].shape[0]

            c1, _, c2, c3 = st.columns([.5, .1, .2, .2])
            c2.metric('Qtd clientes True', value=qtd_true)
            c3.metric('Qtd clientes False', value=len(ypred) - qtd_true)


            # Função para colorir as predições
            def color_pred(val):
                color = 'olive' if val > treshold else 'orangered'
                return f'background-color: {color}'


            tipo_view = st.radio('Visualizar:', ('Completo', 'Apenas predições'))
            if tipo_view == 'Completo':
                df_view = ypred.copy()
            else:
                df_view = pd.DataFrame(ypred.iloc[:, -1].copy())

            st.dataframe(df_view.style.applymap(color_pred, subset=['prediction_score_1']))

            csv = df_view.to_csv(sep=',', decimal=',', index=True)
            st.markdown(f'Shape do CSV a ser baixado: {df_view.shape}')
            st.download_button(label='Download CSV',
                               data=csv,
                               file_name='Predicoes.csv',
                               mime='text/csv')

    else:
        st.warning('Arquivo CSV não foi carregado.')

else:
    st.error('Esta opção será desenvolvida no Checkpoint #2 da disciplina.')
