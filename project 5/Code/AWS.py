#!/usr/bin/env python
# coding: utf-8

# In[7]:


# imports
import numpy as np
import pandas as pd


from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, Normalizer, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer



# In[2]:


# Loading dataset
df = pd.read_csv('../Data/salary_cleaned.csv')
df.head()


# In[3]:


# Converting Year and Month to string
df['year'] = df['year'].astype(str)
df['month'] = df['month'].astype(str)
df.info()


# In[4]:


# Features and target variable
features = ['company', 'title', 'yearsofexperience', 'yearsatcompany', 'year', 'month', 'state_short', 'inflation_rate', 'inflation_rate_3mos', 'employment_rate', 'employment_rate_3mos']

X = df[features]
y = df['totalyearlycompensation']

# Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42)


# In[5]:


# ColumnTransformer
ct = ColumnTransformer([
    ('sc', StandardScaler(), make_column_selector(dtype_exclude=object)),
    ('encoder', OneHotEncoder(handle_unknown='ignore'), ['company', 'title', 'state_short', 'year', 'month'])],
    n_jobs=-1,
    remainder='passthrough')


# In[6]:


X_train_ct = ct.fit_transform(X_train)
X_test_ct = ct.transform(X_test)

X_train_ct.shape, X_test_ct.shape


# In[ ]:


# Instantiate RandomForestRegression
rf = RandomForestRegressor(random_state=42)

# Parameters
rf_params={
    'n_estimators':[100, 200],
    'criterion': ['mse'],
    'max_depth': [2, 4, 6],
    'min_samples_split': [2,4],
    'min_samples_leaf': [1,3]
}

# Instantiating RFR Gridsearch
rf_gs = GridSearchCV(rf, rf_params, cv=5, verbose=1, n_jobs=-1)

# Fitting GridSearch to the data
rf_gs.fit(X_train_ct, y_train)


# In[ ]:


'''
# Train/Test Scores

rf_gs_best_score_train = rf_gs.best_score_
rf_gs_b_score_test = rf_gs.score(X_test_ct, y_test)}

print(f'Best Train Score: {rf_gs.best_score_}')
print(f'Test Score: {rf_gs.score(X_test_ct, y_test)}')

# Best Parameters
rf_gs_best_params = rf_gs.best_params_
print(f'Best parameters: {rf_gs.best_params_}')

# Best Estimators
print(f'Best Estimators: {rf_gs.best_estimator_}')

'''


# In[ ]:


# Predictions
rf_preds_train = rf_gs.predict(X_train_ct)
rf_preds_test = rf_gs.predict(X_test_ct)

# Evaluation
rf_score_train = rf_gs.score(X_train_ct, y_train)
rf_score_test = rf_gs.score(X_test_ct, y_test)

rf_mse_train = metrics.mean_squared_error(y_train, rf_preds_train)
rf_mse_test = metrics.mean_squared_error(y_test, rf_preds_test)

# create the perf evaluation output df
perf_dict = {
    'R2': [rf_score_train, rf_score_test],
    'mse': [rf_mse_train, rf_mse_test]
}

perf_df = pd.DataFrame(perf_dict, index=['train','test']).T

perf_df.to_csv('./randomforest_perf.csv')


# In[ ]:


# create the prediction output df
data_test_copy = X_test.copy()

# Predictions
data_test_copy['totalcomp'] = y_test
data_test_copy['predictions'] = rf_preds_test

data_test_copy.to_csv('./randomforest_pred.csv', index=False)


# In[8]:


# CV results
cv_result = pd.DataFrame(rf_gs.cv_results_)

cv_result.to_csv('./randomforest_CV_Result.csv', index=False)


# In[ ]:
