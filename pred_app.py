import streamlit as st
import pandas as pd
from pycaret.classification import *
from pycaret.regression import *
import subprocess

# Define the command to install packages from requirements.txt
command = ["pip", "install", "-r", "requirements.txt"]

# Title
st.title("PyCaret Prediction App")

# Upload CSV file
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Choose task (classification or regression)
task = st.sidebar.selectbox("Model", ("Classification", "Regression"))

# Load data and select appropriate PyCaret function
if uploaded_file is not None:
    @st.cache
    def load_data():
        data = pd.read_csv(uploaded_file)
        return data

    data = load_data()

    if task == "Classification":
        st.subheader("Classification Model")

        # Add code for classification tasks using PyCaret here
        # Example:
        if st.button("Run Classification"):
            setup(data
                  , target = input_col)
            compare_models()

    elif task == "Regression":
        st.subheader("Regression Model")

        # Add code for regression tasks using PyCaret here
        # Example:
        if st.button("Run Regression"):
            setup(data
                  , target = input_col)
            compare_models()

    st.sidebar.markdown("### Data Sample")
    st.sidebar.write(data.head())
    input_col = st.textarea('Enter your target column')
    input_ex = st.textarea('Enter your exclude column')

else:
    st.warning("Please upload a CSV file.")

# Optional: Add more Streamlit components for user interaction and displaying results.
