from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model and columns
model = joblib.load('crop_yield_model.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    crop_year = int(request.form['crop_year'])
    area = float(request.form['area'])
    production = float(request.form['production'])
    annual_rainfall = float(request.form['annual_rainfall'])
    fertilizer = float(request.form['fertilizer'])
    pesticide = float(request.form['pesticide'])
    crop = request.form['crop']
    season = request.form['season']
    state = request.form['state']

    # Create a DataFrame for the user input
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

    # Convert categorical variables to numerical using one-hot encoding
    user_data_encoded = pd.get_dummies(user_data, columns=['Crop', 'Season', 'State'])

    # Ensure the user input DataFrame has the same columns as the training DataFrame
    user_data_encoded = user_data_encoded.reindex(columns=model_columns, fill_value=0)

    # Make predictions for the user input
    predicted_yield = model.predict(user_data_encoded)[0]

    return render_template('result.html', prediction=predicted_yield)

if __name__ == '__main__':
    app.run(debug=True)
