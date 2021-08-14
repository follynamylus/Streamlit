import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier as GBC, RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression as LoR 
import plotly.express as ex 


st.header("Bees and Pest")
df = pd.read_csv('C:\Streamlit\Bees\intro_bees.csv')
st.selectbox('Period:',['JAN THRU MAR','APR THRU JUN','JUL THRU SEP','OCT THRU DEC'])
st.slider('Percentage of colonies impacted',0.00,100.00)
which_columns = df['State'].unique()
uniques = {i : df['State'].unique() for i in which_columns}
codes = []
for keys,values in uniques.items():
    codes.append(keys)
st.selectbox('States:',codes)
st.selectbox('Year :',['2015','2016','2017','2018','2019'])
st.write(df) 