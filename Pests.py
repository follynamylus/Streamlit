## Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LoR
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,auc,classification_report 
import plotly.express as ex 

## Contents
st.header("Bees and Pest") # Create Header text


df = pd.read_csv('C:\Streamlit\Bees\intro_bees.csv') # <-- Read in Data

# Create Interractive widgets
Period = st.selectbox('Period:',['JAN THRU MAR','APR THRU JUN','JUL THRU SEP','OCT THRU DEC']) # <-- Widget to select through Period categorical feature and save it in period
Percentage_of_colonies_impacted = st.slider('Percentage of colonies impacted',0.00,100.00)# <-- Widget to slide through and save percentage impacted feature
which_columns = df['State'].unique() #<--- Creating a select box option from the dataset
uniques = {i : df['State'].unique() for i in which_columns}
codes = []
for keys,values in uniques.items():
    codes.append(keys)
State = st.selectbox('States:',codes)#<--- Widget to select through and save State categorical column 
year = st.selectbox('Year :',['2015','2016','2017','2018','2019'])#<--Widget to select through and save year categorical column

df['year'] = df['Year'].astype('object')#<-- Convert Year numerical column to categorical

df['Percentage_of_colonies_impacted'] = df['Pct of Colonies Impacted'] #<-- Generate a new column from another

target = [] # Codes to regenerate the target column

for i in df['Affected by']:
    if i == 'Other':
        target.append('Non_pest')
    elif i == 'Pesticides':
        target.append('Non_pest')
    elif i == 'Unknown':
        target.append('Non_pest')
    elif i == 'Pests_excl_Varroa':
        target.append('pest')
    elif i == 'Varroa_mites':
        target.append('pest')
    else :
        target.append('Non_pest')



df['Affected_by'] = target

cols_to_cat = ['Period','State','year']# <-- Columns to dummy

cols_to_drop = ['Period','State','Year','ANSI','state_code','year','Pct of Colonies Impacted','Program']# Columns to drop

train = df.copy() # Copy the dataframe

cats_col = pd.get_dummies(train[cols_to_cat])# Code to dummy the columns

train.drop(cols_to_drop,axis=1,inplace=True) # Code to drop the listed columns

New_train = train.join(cats_col) # Code to join the dummied columns with the dataframe

tags = ['Affected by','Affected_by']

X = New_train.drop(tags,axis=1) # Features for training

y = New_train['Affected_by'] # label for training

Logreg = LoR() # Algorithm to train

Gra = GradientBoostingClassifier(random_state=43) # Algorithm to train

Gra.fit(X,y) # Fitting data to train Gra

Ran = RandomForestClassifier() # Algorithms to train

Ran.fit(X,y) # fitting data to train Ran

Logreg.fit(X, y) # fitting data to train Logreg




data = {'Period' : Period, # codes to convert the stored widget data to dictionary
		'Percentage_of_colonies_impacted' : Percentage_of_colonies_impacted,
		'State' : State,
		'year' : year}

features = pd.DataFrame(data, index=[0]) # Converting the data to dataframe

dummied_features = pd.get_dummies(features) # Dummying the features data

lists = [] # Codes to create a list for the features to predict data so as to have the same column length as the train feature data.

for i in (X.columns) : 
    if i in (dummied_features.columns) :
        lists.append(dummied_features[i].iloc[0])
    else :
        lists.append(0)

pred_lists = np.array([lists]).reshape(1,-1) # Reshaping it to the proper 2 dimentional


st.write(f'Logistic Regression predicts : {Logreg.predict(pred_lists)}') # Predict and output it with logreg


st.write(f'Gradient Boosting predicts : {Gra.predict(pred_lists)}') # Predict and output with Gra

st.write(f'Random Forest predicts : {Ran.predict(pred_lists)}') # Predict and output with Random forest.