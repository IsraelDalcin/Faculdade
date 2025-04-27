#!pip install streamlit
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
# st.subheader('subheader')
# st.write('Write')
# st.markdown('markdown')
# st.info('info')
# st.warning('warning')
# st.error('error')
# st.success('sucesso')
# soma = 23+345
# st.success(soma)

# c1, c2, c3 = st.columns(3)
# c1.warning('warning')
# c2.error('error')
# c3.success('sucesso')

with st.expander('Descrição do App', expanded = False):
    st.write('O objetivo principal deste app .....')

st.sidebar.write('teste sidebar')
with st.sidebar:
    c1, c2 = st.columns([.3, .7])
    c1.image('./images/logo_fiap.png', width = 100)
    c2.write('')
    c2.subheader('Auto ML - Fiap [v1]')

    # database = st.selectbox('Fonte dos dados de entrada (X):', ('CSV', 'Online'))
    database = st.radio('Fonte dos dados de entrada (X):', ('CSV', 'Online'), horizontal = True)
    # st.toggle('Fonte dos dados de entrada (X)')
    # st.checkbox('Fonte dos dados de entrada (X):', value = True)
    # st.selectbox('Fonte dos dados de entrada (X):', ['CSV', 'Online'])
    # st.multiselect('Fonte dos dados de entrada (X):', ['CSV', 'Online'])


    if database == 'CSV':
        st.info('Upload do CSV')
        file = st.file_uploader('Selecione o arquivo CSV', type='csv')

#Tela principal
if database == 'CSV':
    if file:
        #carregamento do CSV
        Xtest = pd.read_csv(file)

        #carregamento / instanciamento do modelo pkl
        mdl_rf = load_model('./pickle/pickle_rf_pycaret2')

        #predict do modelo
        ypred = predict_model(mdl_rf, data = Xtest, raw_score = True)

        with st.expander('Visualizar CSV carregado:', expanded = False):
            c1, _ = st.columns([2,4])
            qtd_linhas = c1.slider('Visualizar quantas linhas do CSV:', 
                                    min_value = 5, 
                                    max_value = Xtest.shape[0], 
                                    step = 10,
                                    value = 5)
            st.dataframe(Xtest.head(qtd_linhas))

        with st.expander('Visualizar Predições:', expanded = True):
            c1, _, c2, c3 = st.columns([.5,.1,.2,.2])
            treshold = c1.slider('Treshold (ponto de corte para considerar predição como True)',
                                min_value = 0.0,
                                max_value = 1.0,
                                step = .1,
                                value = .5)
            qtd_true = ypred.loc[ypred['prediction_score_1'] > treshold].shape[0]

            c2.metric('Qtd clientes True', value = qtd_true)
            c3.metric('Qtd clientes False', value = len(ypred) - qtd_true)
            
            def color_pred(val):
                color = 'olive' if val > treshold else 'orangered'
                return f'background-color: {color}'

            tipo_view = st.radio('', ('Completo', 'Apenas predições'))
            if tipo_view == 'Completo':
                df_view = ypred.copy()
            else:
                df_view = pd.DataFrame(ypred.iloc[:,-1].copy())

            st.dataframe(df_view.style.applymap(color_pred, subset = ['prediction_score_1']))

            csv = df_view.to_csv(sep = ',', decimal = ',', index = True)
            st.markdown(f'Shape do CSV a ser baixado: {df_view.shape}')
            st.download_button(label = 'Download CSV',
                            data = csv,
                            file_name = 'Predicoes.csv',
                            mime = 'text/csv')

        
    else:
        st.warning('Arquivo CSV não foi carregado')
        # st.info('Arquivo CSV não foi carregado')
        # st.error('Arquivo CSV não foi carregado')
        # st.success('Arquivo CSV não foi carregado')

else:
    st.error('Esta opção será desenvolvida no Checkpoint #2 da disciplina')





