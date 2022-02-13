# imports
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, Response, render_template, jsonify

# initialize the flask app
app = Flask('my_app')

# route 1: hello world
# return a simple string
@app.route('/')
def home():
    return "Hello, world!"

# route 2: return a 'web page'
@app.route('/hc_page')
def hc_page():
# return some hard-coded html
    return '<html><body><h1>This is a hard coded page!</h1><p>Here is some hard-coded content. Isn\'t it pretty?</p></body></html>'

# route 3: return some data
# create some data to return as json
# use flask's jsonify function to return the data as well as a 200 status code
@app.route('/the-best')
def the_best():
    best = {
    'food':'tacos',
    'coast':'west',
    'number':42,
    'foobar': None,
    'bool': True
    }
    return jsonify(best)

# route 4: show a form to the user
@app.route('/form')
def form():
    # use flask's render_template function to display an html page
    return render_template('form.html')

# route 5: accept the form submission and do something fancy with it
# load in the form data from the incoming request
# manipulate data into a format that we pass to our model
@app.route('/submit')
def the_prediction():
    user_input = request.args
    X_test = np.array([
    int(user_input['OverallQual']),
    int(user_input['FullBath']),
    int(user_input['GarageArea']),
    int(user_input['LotArea'])
    ]).reshape(1, -1) # turn [1, 2, 3, 4] -> [[1, 2, 3, 4]]
    model = pickle.load(open('assets/model_20211207.p', 'rb')) # 'rb' is read bytes
    pred = model.predict(X_test)[0]
    pred = round(pred, 2)

    '''if pred > 1000000:
        pred = 'You have a baller pad'
    else:
        pred
'''
    # return jsonify({'pred': pred})
    return render_template('results.html', prediction = pred)















# Call app.run(debug=True) when python script is called
if __name__ == '__main__':
    app.run(debug = True)
