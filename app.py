import streamlit as st
import pandas as pd
import os

#pandas profiling 
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

#for ML
from pycaret.classification import setup, compare_models, pull, save_model

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv",index_col=None)

with st.sidebar:
    st.image("Robot.png")
    st.write("")
    st.markdown("<h2 style='text-align: center'>Choose Your Step</h2>",unsafe_allow_html=True)
    col1, col2 = st.columns([1,3])
    with col1:
        st.write(' ')
    with col2:
        choice = st.radio("",["Upload Dataset","Profiling","Learning","Download"])
    st.write("")
    st.info("This application helps you create and download your own machine learning model with zero coding experience.")


if choice == "Upload Dataset":
    st.markdown("<h1 style='text-align: center; color: white; padding:'>Create Your Machine Learning Model</h1>", unsafe_allow_html=True)
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.markdown("<h1 style='text-align: center; color: white; padding:'>Here is the Analysis of Your Dataset</h1>", unsafe_allow_html=True)
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice == "Learning":
    st.markdown("<h1 style='text-align: center; color: white; padding:'>Now Let's Train Our Model</h1>", unsafe_allow_html=True)
    chosen_target = st.selectbox("Choose the Target Variable:",df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == "Download":
    st.markdown("<h1 style='text-align: center; color: white; padding:'>Let's Download Our Machine Learning Model</h1>", unsafe_allow_html=True)
    st.write("")
    st.markdown("<p style='text-align: center; color: white; padding:'>Here is your fully capable machine learning model. This model has been picked out from a list of algorithms. We have run all algorithms to see which performs best for the given dataset and hence provided you with the optimal machine learning model!</p>", unsafe_allow_html=True)
    st.write("")
    with open('best_model.pkl', 'rb') as f: 
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write('')
        with col2:
            st.download_button('Download Model', f, file_name="best_model.pkl")
        with col3:
            st.write('')