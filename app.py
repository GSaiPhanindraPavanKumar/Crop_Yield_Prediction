from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the model and columns
model = joblib.load('crop_yield_model.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    crop_year = int(request.form['crop_year'])
    area = float(request.form['area'])
    production = float(request.form['production'])
    annual_rainfall = float(request.form['annual_rainfall'])
    fertilizer = float(request.form['fertilizer'])
    pesticide = float(request.form['pesticide'])
    crop = request.form['crop']
    season = request.form['season']
    state = request.form['state']

    user_data = pd.DataFrame({
        'Crop_Year': [crop_year],
        'Area': [area],
        'Production': [production],
        'Annual_Rainfall': [annual_rainfall],
        'Fertilizer': [fertilizer],
        'Pesticide': [pesticide],
        'Crop': [crop],
        'Season': [season],
        'State': [state]
    })

    user_data_encoded = pd.get_dummies(user_data, columns=['Crop', 'Season', 'State'])
    user_data_encoded = user_data_encoded.reindex(columns=model_columns, fill_value=0)
    predicted_yield = model.predict(user_data_encoded)[0]

    return render_template('result.html', prediction=predicted_yield)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
