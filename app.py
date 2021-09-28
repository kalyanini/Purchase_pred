import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('Purchaser_pred_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [np.array(list(x)).astype(int) for x in request.form.values()]
    prediction = model.predict(int_features)

    output = round(prediction[0], 2)
    output = 'Yes' if output == 1 else 'No'
    return render_template('index.html', prediction_text='Purchaser? {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls through request
    '''
    
    prediction = model.predict([np.array(list(data.values[0])).astype(int)])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)