# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Quiz 3 Retake

## Task
Your task for quiz 3 is to write a **python script** (.py file) that:
- Imports the libraries you need
- Reads in the data: `data/diabetes.csv`
- Processes data:
    1. Set the `patient_id` column as the index
    2. Converts the `target_diabetes` column to 0/1 using 0 for 'Negative' and 1 for 'Positive'
- Models data using any classification algorithm you would like
    - If you do not know what to do here, feel free to fit a Logistic Regression model with default hyperparameters. But you are free to use any other model or methods that you know if you think that would be better!
    - Use a random state of 42 when splitting your data into training and testing
    - Use ALL columns (except your target) as your X matrix
- Generates predictions on your test data
- Creates a new DataFrame that has:
    1. One column called `predictions` which is the predictions from your model on your test data
    2. An index that is the id of the patient associated with the prediction from your test data
- Writes the DataFrame with the predictions to a csv called `predictions.csv` in the data folder in this repository

This script should be able to be run in your terminal: `python 03-quiz-retake.py`

**Note**: to get as many points as possible on this quiz, your submission for this quiz should be a .py file **not** a .ipynb file.

The dataset provided is from [here](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv). The goal is to predict the onset of diabetes in patients (0 for a negative diabetes test and 1 for a positive diabetes test). The data dictionary for the dataset is below:

Variable Name | Description
--- | ---							
pregnant | Number of times pregnant
plasma | Plasma glucose concentration a 2 hours in an oral glucose tolerance test
bp | Diastolic blood pressure (mm Hg)
triceps | Triceps skin fold thickness (mm)
serum | 2-Hour serum insulin (mu U/ml)
bmi | Body mass index (weight in kg/(height in m)^2)
diabetes_pedigree | Diabetes pedigree function
age | Age (years)
patient_id | Fictional Patient ID number
target_diabetes | Target variable: Positive if diabetes test is positive, Negative if diabetes test is negative

The .py file in this repository contains the instructions mentioned above. You are to clone this repository and fill in the quiz with your answers, taking special care to follow the directions _exactly_.

To **submit** your quiz, drop your completed .py script in the file uploader in the provided Google Form. Note that your submission for this quiz should be a .py file **not** a .ipynb file.

## Quiz Information
- This is an "open book" quiz - you may use any resources! This includes Jupyter notebooks, Google, StackOverflow, and your notes. ANY RESOURCE YOU USE MUST BE CITED (comment your code with links, which lesson you got the inspiration from, etc.) and failure to cite resources you use will be considered plagiarism.
- You may NOT work with anyone else on this quiz (your classmates, your instructor, etc.).
- You will have one hour to take this quiz.
- Do your best and good luck!

## Learning Targets Assessed
You should be able to:
- Perform basic data cleaning and EDA
- Fit, generate predictions from, and evaluate a classification model

## Motivation
In many data science interviews, you will be required to complete a technical test or take-home assignment. These are often in the form of a coding challenge or a task given for you to complete within a certain amount of time. These quizzes are designed to mimic that process to prepare you to be successful during the interview process.
