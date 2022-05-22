from turtle import width
from numpy import diff
import streamlit as st
import pickle
import pandas as pd

#Extraer los archivos pickle
with open('lin_reg.pkl', 'rb') as li:
     lin_reg = pickle.load(li)

with open('log_reg.pkl', 'rb') as lo:
     log_reg = pickle.load(lo)

with open('svc_m.pkl', 'rb') as sv:
     svc_m = pickle.load(sv)

def classify(num):
    if num == 0:
        return 'Setosa'
    elif num == 1: 
        return 'Versicolor'
    else:
        return 'Virginica'

def main ():
    st.title('Mi primer modelado de ML con Iris by @Sebas_Barragan')
    st.sidebar.header('User imput Parameters')

    def user_imput_parameters():
        sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
        sepal_width = st.sidebar.slider('Sepal width', 2.00, 4.4, 3.4)
        petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
        petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
        data = {'sepal_length': sepal_length,
                'sepal_width' : sepal_width, 
                'petal_length': petal_length,
                'petal_width' : petal_width,
                 }
        features = pd.DataFrame(data, index=[0])
        return features

    df= user_imput_parameters()

    option = ['Linear Rregression', 'Logistic Regression', 'SVM']
    model = st.sidebar.selectbox('which model you like to use? irina', option)

    st.subheader('User Imput Parameters')
    st.subheader(model)
    st.write(df)
    
    if st.button('run'):
        if model == 'Linear Rregression':
            st.success(classify(lin_reg.predict(df)))
        elif model == 'Logistic Regression':
            st.success(classify(log_reg.predict(df)))
        else:
            st.success(classify(svc_m.predict(df)))


if __name__ == '__main__':
    main()