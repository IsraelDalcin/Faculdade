import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pycaret.classification import * #load_model, predict_model

st.set_page_config( page_title = 'Simulador - Case Ifood',
                    page_icon = './images/logo_fiap.png',
                    layout = 'wide',
                    initial_sidebar_state = 'expanded')

st.title('Simulador - Conversão de Vendas')

with st.expander('Descrição do App', expanded = False):
    st.write('O objetivo principal deste app .....')
st.sidebar.write('teste sidebar')

with st.sidebar:
    c1, c2 = st.columns([.3, .7])
    c1.image('./images/logo_fiap.png', width = 100)
    c2.write('')
    c2.subheader('Auto ML - Fiap [v1]')

    database = st.radio('Fonte dos dados de entrada (X):', ('CSV', 'Online'), horizontal = True)
    
    if database == 'CSV':
        st.info('Upload do CSV')
        file = st.file_uploader('Selecione o arquivo CSV', type='csv')

if database == 'CSV':
    if file:
        #carregamento do CSV
        Xtest = pd.read_csv(file)

        #carregamento / instanciamento do modelo pkl
        mdl_rf = load_model('./pickle/pickle_rf_pycaret2')

        #predict do modelo
        #ypred = predict_model(mdl_rf, data = Xtest, raw_score = True)

        with st.expander('Visualizar CSV carregado:', expanded = False):
            c1, _ = st.columns([2,4])
            qtd_linhas = c1.slider('Visualizar quantas linhas do CSV:', 
                                    min_value = 5, 
                                    max_value = Xtest.shape[0], 
                                    step = 10,
                                    value = 5)
            st.dataframe(Xtest.head(qtd_linhas))