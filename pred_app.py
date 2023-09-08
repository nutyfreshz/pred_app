import streamlit as st
import pandas as pd

# Title
st.title("PyCaret Prediction App")

# Upload CSV file
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Choose task (classification or regression)
task = st.sidebar.selectbox("Model", ("Classification", "Regression"))

if uploaded_file is not None:
    @st.cache
    def load_data():
        data = pd.read_csv(uploaded_file)
        return data

    data = load_data()
    st.sidebar.markdown("### Data Sample")
    st.sidebar.write(data.head())

    # Define functions
    def run_pycaret(data, task, input_col, input_ex):
        if task == "Classification":
            st.subheader("Classification Model")

            from pycaret.regression import *
            
            if st.button("Run Classification"):
                setup(data, target=input_col, ignore_features=input_ex)
                compare_models()

        elif task == "Regression":
            st.subheader("Regression Model")

            from pycaret.classification import *
            
            if st.button("Run Regression"):
                setup(data, target=input_col, ignore_features=input_ex)
                compare_models()

    input_col = st.sidebar.text_area('Enter your target column')
    input_ex = st.sidebar.text_area('Enter your exclude column')

    run_pycaret(data, task, input_col, input_ex)
else:
    st.warning("Please upload a CSV file.")
