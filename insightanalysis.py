from contextlib import nullcontext
from ssl import Options
import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px  # pip install plotly-express
import streamlit as st  # pip install streamlit
from PIL import Image
from scipy.stats import shapiro
from scipy.stats import pearsonr
from scipy.stats import f_oneway


st.set_page_config(page_title="insight analyzer",
                   page_icon=":bar_chart:", layout="wide")

st.title(":bar_chart: Insight analysis ")
st.write("Drop your data to us ...........")


uploaded_file = st.file_uploader("Upload a file here")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)

    df_types = pd.DataFrame(df.dtypes, columns=['Data Type'])
    numerical_cols = df_types[~df_types['Data Type'].isin(
        ['int'])].index.values

    st.write("COLUMNS Available are :", numerical_cols)
    st.write("Numerical columns are", df.select_dtypes(include=np.number))
    st.write("Description values in each columns:", df.describe())

    tab1, tab2, tab3 = st.tabs(["Max", "Min", "Average"])

    with tab1:
        st.header("Maximum")
        st.write("Maximum values in each columns:", df.max(numeric_only=True))

    with tab2:
        st.header("Minimum")
        st.write("Minimum values in each columns:", df.min(numeric_only=True))

    with tab3:
        st.header("Average")
        st.write("Average values in each columns:", df.mean(numeric_only=True))

    fig, ax = plt.subplots()
    sn.heatmap(df.corr(), ax=ax)

    st.write("Statistical Tests are :")
    name = st.text_input(
        'Enter a column name to continue with statistical tests: ',)
    st.write("Sharpiro Test:")
    nor_test = df[name].tolist()
    stat, p = shapiro(nor_test)
    st.write('stat=%.3f, p=%.5f' % (stat, p))
    if p > 0.05:
        st.write('Probably Gaussian')
    else:
        st.write('Probably not Gaussian')

    st.write("Statistical Tests are :")
    dep1 = st.text_input(
        'Enter a depandant column name : ',)
    dep2 = st.text_input(
        'Enter a independant column name : ',)
    st.write("Pearsons Correlation Test:")
    per_test1 = df[dep1].tolist()
    per_test2 = df[dep2].tolist()
    stat, p = pearsonr(per_test1, per_test2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably independent')
    else:
        print('Probably dependent')
    st.write("Heatmap", fig)

st.sidebar.header("look what we got")
st.sidebar.write("blah blah \n blah")
st.sidebar.write("About us")
st.sidebar.write("blah blah \n blah")
st.sidebar.image("https://static.streamlit.io/examples/cat.jpg", width=200)