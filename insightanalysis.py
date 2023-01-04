import base64
from contextlib import nullcontext
from queue import Full
from ssl import Options
import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
import streamlit as st
from PIL import Image
from scipy.stats import shapiro
from scipy.stats import pearsonr
from scipy.stats import f_oneway
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats import chi2_contingency
import scipy.stats as stats
from sklearn.ensemble import ExtraTreesRegressor
import time
import mysql.connector
from collections.abc import Iterable


st.set_page_config(page_title="Data Dumper",
                   page_icon=":bar_chart:", layout="wide")
                   
st.markdown("<h1 style='text-align: center; color: grey;'>ANALYSE THIS!</h1>", unsafe_allow_html=True)



def home():
    st.image("insight.png",
         width=1000)
    st.markdown("""
<style>
body {
  background: #ff0099; 
  background: -webkit-linear-gradient(to right, #ff0099, #493240); 
  background: linear-gradient(to right, #ff0099, #493240); 
}
</style>
    """, unsafe_allow_html=True)
    st.write(
        "Without big data analytics,Companies are blind and deaf, wandering out onto the web like a deer on the freeway -Geoffrey Moore")
    st.write("Writing up the results of a data analysis is not a skill that anyone is born with. It requires practice and, at least in the beginning, a bit of guidance.  Thus we provide easy analysis for the users without the knowledge of coding so that they can infer knowledge from existing data.")
    st.write("Data Is the future and the future is now!Every mouse click,keyboard button press,swipe or tap is used to shape business decisions. Everything is about data these days. Data is information and Information is power")
    st.write(" Contact")
    st.write(":telephone_receiver: 9444967878")
    st.write(":e-mail: contact@nikrundevleopers.com")
    st.write(":iphone: www.linkedin.com/in/nikrundevelopers")
    st.write(":round_pushpin: Coimbatore,India")
    

uploaded_file = st.file_uploader("Upload a file here")
if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)


def upload(): 
# with st.spinner('Wait for it...'):
       # time.sleep(2)  
 global df
 if df.empty:
    st.write( "Enter appropriate values")
 else:
        st.write("The given data is... :")
        st.write(df)

            # cleaning Data

        st.write("The checking null values from the given data is...:")
        st.write(df.isnull().sum())
        n = df.fillna(df.mean())
        st.write("After filling null values ", df.fillna(df.mean()))
        st.write("Click to download file")
        def get_table_download_link_csv(n):
                csv = n.to_csv().encode()
                b64 = base64.b64encode(csv).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="fillednull.csv" target="_blank">Download csv file with filled null values</a>'
                return href
        st.markdown(get_table_download_link_csv(n), unsafe_allow_html=True)

        
        st.write("If the DataSet contains numerical data, the description contains these information for each column:  count - The number of not-empty values. mean - The average (mean) value. std - The standard deviation. min - the minimum value. 25% - The 25% percentile. 50% - The 50% percentile. 75% - The 75% percentile. max - the maximum value.")

        st.write("Description values in each columns:", df.describe())

        st.write(
                "The Maximum minimum and average values in each columns are :")

        tab1, tab2, tab3 = st.tabs(["Max", "Min", "Average"])

        with tab1:
            #encode data
                label_encoders = {}
                categorical_columns = df.columns
                for columns in categorical_columns:
                    label_encoders[columns] = LabelEncoder()
                    df[columns] = label_encoders[columns].fit_transform(df[columns])

                st.header("Maximum")
                st.write("Maximum values in each columns:", df.max(numeric_only=True))

        with tab2:
                st.header("Minimum")
                st.write("Minimum values in each columns:", df.min(numeric_only=True))

        with tab3:
                st.header("Average")
                st.write("Average values in each columns:", df.mean(numeric_only=True))

        tab1, tab2, tab3 = st.tabs(["Heatmap", "Distplot", "Piechart"])

        with tab1:
                fig, ax = plt.subplots(figsize=(3, 3))
                sn.heatmap(df.corr(), ax=ax)
                st.write("A heatmap ( or heat map) is a graphical representation of data where values are depicted by color./n  They are essential in detecting what does or doesn't work on a website or product page. By experimenting with how certain buttons and elements are positioned on your website, heatmaps allow you to evaluate your product’s performance and increase user engagement and retention as you prioritize the jobs to be done that boost customer value. Heatmaps make it easy to visualize complex data and understand it at a glance:")
                st.pyplot(fig)
        with tab2:
                d5 = st.text_input('Enter column name for plotting : ')
                if d5 == "":
                    st.write("Enter appropriate values")
                plot = df[d5].values
                progress_bar = st.sidebar.progress(0)
                st.sidebar.write("Plotting...")
                status_text = st.sidebar.empty()
                last_rows = np.random.randn(1, 1)
                chart = st.line_chart(plot)

                for i in range(1, 101):
                    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
                    status_text.text("%i%% Complete" % i)
                    chart.add_rows(new_rows)
                    progress_bar.progress(i)
                    last_rows = new_rows
                    time.sleep(0.05)
                progress_bar.empty()
def mlr():
# with st.spinner('Wait for it...'):
       # time.sleep(2)
 global df
 if df.empty:
    st.write( "Enter appropriate values")
 else:
    label_encoders = {}
    categorical_columns = df.columns
    for columns in categorical_columns:
            label_encoders[columns] = LabelEncoder()
            df[columns] = label_encoders[columns].fit_transform(df[columns])

    st.markdown(f"# {list(page_names_to_funcs.keys())[2]}")
    tab4,tab5= st.tabs(["Linear Regression","Multiple Linear Regression"])
    with tab4:
                st.write("The given data is... :")
                st.write(df)
                d = st.text_input('Enter a Dependant : ')
                if d == "":
                    st.write("Enter appropriate values")
                else:
                # train split
                # split a dataset into train and test sets
                    from sklearn.datasets import make_blobs
                    from sklearn.model_selection import train_test_split
                    
                    X = df.drop([d], axis=1).values
                    y = df[d].values
            # create dataset
            # split into train test sets
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
                    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
                    from sklearn.linear_model import LinearRegression
                    ml = LinearRegression()
                    ml.fit(X_train, y_train)
                    y_pred = ml.predict(X_test)

                    from sklearn.metrics import r2_score
                    st.write("R2 Score")
                    st.write(r2_score(y_test, y_pred))
                    st.write("Root Mean Square Error")

                    rms = sqrt(mean_squared_error(y_test, y_pred))
                    st.write(rms)

                    pred_y = pd.DataFrame(
                        {'Actual Value': y_test, 'Predicted value': y_pred, 'difference': y_test-y_pred})
                    st.write(pred_y[0:20])

                #from sklearn.externals import joblib

                #joblib.dump(ml, 'ml.pkl')
    def get_table_download_link_csv(data):
                csv = data.to_csv().encode()
                b64 = base64.b64encode(csv).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="ml.pkl" target="_blank">Download .pkl file</a>'
                return href
    st.markdown(get_table_download_link_csv(df), unsafe_allow_html=True)
    with tab5:
        st.write("The given data is... :")
        st.write(df)
        if df.empty:
            st.write( "Enter appropriate values")
        else:
            label_encoders = {}
            categorical_columns = df.columns
            for columns in categorical_columns:
                    label_encoders[columns] = LabelEncoder()
                    df[columns] = label_encoders[columns].fit_transform(df[columns])

            st.markdown(f"# {list(page_names_to_funcs.keys())[2]}")
            d1 = st.text_input('Enter required value : ')
            if d1 == "":
                st.write( "Enter appropriate values")
            f1 = st.text_input('Enter feature column 1: ')
            if f1 == "":
                st.write( "Enter appropriate values")
            f2 = st.text_input('Enter feature column 2: ')
            if f2 == "":
                st.write( "Enter appropriate values")
            else:
                X = df[[f1, f2]]
                y = df[d1]
                regr = linear_model.LinearRegression()
                regr.fit(X, y)
                st.write(regr.coef_)
                val1 = st.number_input('Enter Value 1 for Prediction: ')
                if val1 == "":
                    st.write( "Enter appropriate values")
                val2 = st.number_input('Enter Value 2 for Prediction: ')
                if val2 == "":
                    st.write( "Enter appropriate values")
                predictedCO2 = regr.predict([[val1, val2]])
                st.write("Predicted Value: ")
                st.write(predictedCO2)
                
def feature():
    # with st.spinner('Wait for it...'):
       # time.sleep(2)
    st.markdown(f'# {list(page_names_to_funcs.keys())[4]}')
    global df
    st.write(df)
    if df.empty:
        st.write( "Enter appropriate values")
    else:
        label_encoders = {}
        categorical_columns = df.columns
        for columns in categorical_columns:
                    label_encoders[columns] = LabelEncoder()
                    df[columns] = label_encoders[columns].fit_transform(df[columns])
        d2 = st.text_input('Enter a required : ')
        if d2 == "":
                st.write("Enter feature value",tooltips=list)
        else:    
                        X = df.drop([d2], axis=1).values
                        y = df[d2]
                        model = ExtraTreesRegressor()
                        model.fit(X,y)
                        st.write(model.feature_importances_)
                        st.info('The highest value column has higher dependencies in the feature variable', icon="ℹ️")


page_names_to_funcs = {
    "Main": home,
    "Data PreProcessing":upload,
    "Regression and Prediction": mlr,
    "Feature Extraction": feature
}


demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
