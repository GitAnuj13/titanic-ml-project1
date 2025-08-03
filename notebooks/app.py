from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and preprocessor
model = joblib.load('titanic_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form values
    features = [float(x) for x in request.form.values()]
    
    # Convert to dataframe
    input_df = pd.DataFrame([features], columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','FareBand'])
    
    # Preprocess input
    #X_processed = preprocessor.transform(input_df)

    # Predict
    prediction = model.predict(input_df)
    
    return render_template('index.html', prediction_text=f'Survival Prediction: {"Survived" if prediction[0]==1 else "Did Not Survive"}')

if __name__ == '__main__':
    app.run(debug=True)
