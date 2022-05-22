import sys

from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np

# Your API definition
app = Flask(__name__)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

str = 'admiration	amusement	anger	annoyance	approval	caring	confusion	curiosity	desire	disappointment	disapproval	disgust	embarrassment	excitement	fear	gratitude	grief	joy	love	nervousness	optimism	pride	realization	relief	remorse	sadness	surprise	neutral'

emotions_list = str.split('\t')



@app.route('/predict', methods=['POST'])
def predict():
    if lr:
        try:
            json_ = request.json

            json_str = json_[0]['text']


            prediction = lr.predict(vectorizer.transform([json_str]))

            prediction_emotion = emotions_list[int(prediction) - 1]

            return jsonify({'prediction': prediction_emotion})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    lr = joblib.load("model.pkl") # Load "model.pkl"
    vectorizer = joblib.load('vectorizer.pkl')
    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    app.run(port=port, debug=True)