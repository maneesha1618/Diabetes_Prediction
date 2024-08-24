from flask import Flask,render_template,request
from  sklearn.preprocessing import StandardScaler,LabelEncoder
import pickle
import numpy as np

with open('label.pkl','rb') as f:
    label_encoder=pickle.load(f)

with open('scalar.pkl','rb') as f:
    scale=pickle.load(f)

model=pickle.load(open('adb_model.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def homepage():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home1():
    try:
        # Retrieve and convert inputs
        age = float(request.form['a'])
        hypertension = float(request.form['b'])
        heart_disease = float(request.form['c'])
        smoking_history = request.form['d']  # Expecting a string value
        Bmi = float(request.form['e'])
        Hba1c_level = float(request.form['f'])
        blood_glucose_level = float(request.form['g'])

        # Encode smoking history
        try:
            smoking_history_encoded = label_encoder.transform([smoking_history])[0]
        except ValueError:
            return render_template('error.html', message="Invalid smoking history value")

        # Create the input array for the model
        arr = np.array([[age, hypertension, heart_disease, smoking_history_encoded, Bmi, Hba1c_level, blood_glucose_level]])

        # Scale the input data
        input_data_scaled = scale.transform(arr)

        # Predict
        pred = model.predict(input_data_scaled)

        # Render the result
        return render_template('after.html', data=pred[0])
    except Exception as e:
        return render_template('error.html', message=str(e))

if __name__=="__main__":
    app.run(debug=True)