import pandas as pd
import pickle
import streamlit as st

df = pd.read_csv('../Data/salary_cleaned.csv')

with open('../Code/svr.pkl', 'rb') as f:
    svr = pickle.load(f)

#with open('../Models/ColumnTransformer.pkl', 'rb') as f:
    #ct_pkl = pickle.load(f)

year = 2020
month = 9
inflation_rate = 0.014
inflation_rate_3mos = 0.006
employment_rate = 0.93582054649934
employment_rate_3mos = 0.8939803405002691

st.sidebar.header('Total compensation Predictor')
select = st.sidebar.selectbox('Select Form', ['Form 1'], key='1')
if not st.sidebar.checkbox("Hide", True, key='1'):
    st.title('Fill in the following information:')
    company = st.selectbox(label = 'Company name:', options = df['company'].unique())
    title = st.selectbox(label = 'Position title:', options = df['title'].unique())
    yearsofexperience = st.number_input("Years of experience:")
    yearsatcompany =  st.number_input("Years at company:")
    state = st.selectbox(label = 'US State:', options = df['state_short'].unique())
submit = st.button('Predict')

#new_data = {
    #'company': company,
    #'title': title,
    #'yearsofexperience': yearsofexperience,
    #'yearsatcompany': yearsatcompany,
    #'year': str(year),
    #'month': str(month),
    #'state': state,
    #'inflation_rate_3mos': inflation_rate_3mos,
    #'employment_rate': employment_rate,
    #'employment_rate_3mos': employment_rate_3mos
#}

#if submit:

    #ew_data_df = pd.DataFrame(new_data, index=[0])


    #new_dummy = pd.get_dummies(new_data_df, columns = ['company', 'title', 'state_short'], drop_first = True)
    #new_data_df_ct = sc.transform(new_dummy)

    #prediction = svr.predict(new_data_df_ct)
    #prediction
