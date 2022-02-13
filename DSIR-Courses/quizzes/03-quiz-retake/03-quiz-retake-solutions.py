#!/usr/bin/env python3

"""
Quiz 3 - Sample Solutions
Version 2

Task:
Your task for quiz 3 is to write a python script that:
- Imports the libraries you need
- Reads in the data: data/diabetes.csv
- Processes data:
    1. Set the 'patient_id' column as the index
    2. Converts the 'target_diabetes' column to 0/1 using 0 for 'Negative' and 1 for 'Positive'
- Models data using any classification algorithm you would like
    - If you do not know what to do here, feel free to fit a Logistic Regression model with default hyperparameters. But you are free to use any other model or methods that you know if you think that would be better!
    - Use a random state of 42 when splitting your data into training and testing
    - Use ALL columns (except your target) as your X matrix
- Generates predictions on your test data
- Creates a new DataFrame that has:
    1. One column called 'predictions' which is the predictions from your model on your test data
    2. An index that is the id of the patient associated with the prediction from your test data
- Writes the DataFrame with the predictions to a csv called 'predictions.csv' in the data folder in this repository

This script should be able to be run in your terminal: python 03-quiz-retake.py

Rubric (17 points total):
+ 1 : for importing libraries
+ 1 : for reading in the data
+ 1 : for setting patient_id column as the index
+ 1 : for converting target_diabetes column to 0/1
+ 1 : for correctly setting up X and y
+ 1 : for splitting data into training & testing
+ 1 : for correctly instantiating a classification model
+ 1 : for correctly fitting a classification model
+ 1 : for generating predictions on test data
+ 1 : for creating a DataFrame for the predictions (full credit will be given if a student uses methods, i.e. numpy array, here too)
+ 1 : for including the patient id as the index in predictions DataFrame (full credit will also be given for including the ids as another column)
+ 1 : for including a column called predictions in predictions DataFrame (points will not be deducted for other column names used)
+ 1 : for writing a csv called predictions.csv (points will not be deducted for other csv names used)
+ 1 : for writing the csv in the data folder
+ 3 : for submitting a python script rather than a Jupyter notebook

"""

# Import the libraries you need
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Read in the data: data/diabetes.csv
df = pd.read_csv('./data/diabetes.csv')

# Process data:
# 1. Set the 'patient_id' column as the index
df.set_index('patient_id', inplace=True)

# 2. Convert the 'target_diabetes' column to 0/1 using 0 for 'Negative' and 1 for 'Positive'
df['target_diabetes'] = df['target_diabetes'].map({'Negative': 0, 'Positive': 1})

# Model data using any classification algorithm you would like
# Use a random state of 42 when splitting your data into training and testing
X = df.drop(columns = 'target_diabetes')
y = df['target_diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

lr = LogisticRegression(solver = 'liblinear')

lr.fit(X_train, y_train)

# Generate predictions on your test data
preds = lr.predict(X_test)

# Create a new DataFrame for predictions
pred_df = pd.DataFrame()

# 2. Have one column called 'predictions'
# which is the predictions from your model on your test data
pred_df['predictions'] = preds

# 1. Have an index that is the id of the patient
pred_df.index = y_test.index

# Write the DataFrame you created to a csv called 'predictions.csv'
# in the data folder in this repository
pred_df.to_csv('./data/predictions.csv', index = True)
