# pip install flask

from flask import Flask, render_template, request
import numpy as np
import pickle

with open('house_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['post'])
def predict():
    bed = int(request.form['bedrooms'])
    bath = int(request.form['bathrooms'])
    loc = int(request.form['location'])
    size = int(request.form['size'])
    status = int(request.form['status'])
    face = int(request.form['facing'])
    Type = int(request.form['type'])
    
    input_data = np.array([[bed, bath, loc, size, status, face, Type]])
    
    predicted_price = model.predict(input_data)[0]
    
    return render_template('index.html', predicted_price = predicted_price)

if '__name__' == '__main__':
    app.run()
