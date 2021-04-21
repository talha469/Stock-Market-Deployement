import numpy as np
from pmdarima import auto_arima
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('ARIMAStockPrediction.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(n_periods = 1, exogenous = final_features)
    output = float(prediction)

    return render_template('index.html', prediction_text='Volume Weighted Average Price Will be : {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)