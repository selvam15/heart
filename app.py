from Flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import requests

app = Flask(__name__)

heart_data = pd.read_csv(r'\heart_disease_data.csv')

# Splitting the Features and Target
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

model = LogisticRegression()
model.fit(X, Y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form.get('age'))
    sex = int(request.form.get('sex'))
    cp = int(request.form.get('cp'))
    trestbps = int(request.form.get('trestbps'))
    chol = int(request.form.get('chol'))
    fbs = int(request.form.get('fbs'))
    restecg = int(request.form.get('restecg'))
    thalach = int(request.form.get('thalach'))
    exang = int(request.form.get('exang'))
    oldpeak = int(request.form.get('oldpeak'))
    slope = int(request.form.get('slope'))
    ca = int(request.form.get('ca'))
    thal = int(request.form.get('thal'))

   
        
    input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)

    if prediction[0] == 0:
        res1 = 'The Person does not have a Heart Disease'
        res2 = ' You Are Healthy'
    else:
        res1 = 'The Person has Heart Disease'
        res2 = 'Please go and Consult a Doctor'

    return render_template('index.html', res1=res1, res2=res2)


@app.route('/button_click', methods=['POST'])
def button_click():
    print("Button clicked!")
    return 'Button clicked!'

@app.route('/styles.css')
def serve_css():
    return app.send_static_file('styles.css')

@app.route('/scripts.js')
def serve_js():
    return app.send_static_file('scripts.js')

@app.route('/signup')
def signup():
    return render_template('signup.html')

if __name__ == '__main__':
    app.run(debug=True)

