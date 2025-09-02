# Crop Yield Prediction Web Application

A machine learning web application built with Flask that predicts crop yields based on various agricultural parameters. The application uses a Linear Regression model trained on historical crop data to provide accurate yield predictions for different crops, seasons, and states.

## 🌾 Features

- **Crop Yield Prediction**: Predict yield based on multiple input parameters
- **Interactive Web Interface**: User-friendly form-based input system
- **Multi-Parameter Analysis**: Consider various factors affecting crop yield:
  - Crop Year
  - Area (in hectares)
  - Production (in tonnes)
  - Annual Rainfall (in mm)
  - Fertilizer usage (in kg/hectare)
  - Pesticide usage (in kg/hectare)
  - Crop Type
  - Season
  - State
- **Real-time Predictions**: Get instant yield predictions through the web interface
- **Pre-trained Model**: Uses a Linear Regression model trained on historical data

## 🛠️ Technology Stack

- **Backend**: Flask (Python web framework)
- **Machine Learning**: scikit-learn (Linear Regression)
- **Data Processing**: pandas, numpy
- **Model Persistence**: joblib
- **Frontend**: HTML, CSS (responsive design)
- **Development**: Python 3.x

## 📁 Project Structure

```
Crop_Yield_Prediction/
├── Main.py                 # Model training script
├── app.py                  # Flask web application
├── crop_yield.csv          # Training dataset
├── crop_yield_model.pkl    # Trained model file
├── model_columns.pkl       # Model feature columns
├── requirements.txt        # Python dependencies
├── index.html             # Standalone HTML (if needed)
├── templates/             # Flask templates
│   ├── index.html         # Main input form
│   └── result.html        # Prediction results page
└── static/                # CSS styles and assets
    └── styles.css         # Application styling
```

## ⚙️ Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/GSaiPhanindraPavanKumar/Crop_Yield_Prediction.git
   cd Crop_Yield_Prediction
   ```

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install individually:
   ```bash
   pip install Flask==2.3.2
   pip install pandas==2.0.0
   pip install scikit-learn==1.2.0
   pip install joblib==1.3.2
   pip install numpy==1.24.2
   ```

3. **Verify model files exist**:
   - Ensure `crop_yield_model.pkl` and `model_columns.pkl` are present
   - If not, run the training script first (see Training section below)

## 🚀 Usage

### Running the Web Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Access the application**:
   - Open your web browser
   - Navigate to `http://localhost:5000` or `http://127.0.0.1:5000`

3. **Make predictions**:
   - Fill in the required parameters in the web form
   - Select crop type, season, and state from dropdown menus
   - Enter numerical values for year, area, production, rainfall, fertilizer, and pesticide
   - Click submit to get the yield prediction

### Training the Model (Optional)

If you want to retrain the model with new data:

1. **Update the dataset**:
   - Replace or modify `crop_yield.csv` with your data
   - Ensure the CSV has the required columns

2. **Run the training script**:
   ```bash
   python Main.py
   ```

3. **Generated files**:
   - `crop_yield_model.pkl`: Trained Linear Regression model
   - `model_columns.pkl`: Feature column names for consistency

## 📊 Model Details

### Algorithm
- **Model Type**: Linear Regression
- **Library**: scikit-learn
- **Features**: 9 input parameters (6 numerical + 3 categorical)
- **Encoding**: One-hot encoding for categorical variables

### Input Features
1. **Numerical Features**:
   - Crop_Year: Year of cultivation
   - Area: Area under cultivation (hectares)
   - Production: Total production (tonnes)
   - Annual_Rainfall: Yearly rainfall (mm)
   - Fertilizer: Fertilizer usage (kg/hectare)
   - Pesticide: Pesticide usage (kg/hectare)

2. **Categorical Features**:
   - Crop: Type of crop (encoded using one-hot encoding)
   - Season: Growing season (encoded using one-hot encoding)
   - State: Indian state (encoded using one-hot encoding)

### Target Variable
- **Yield**: Crop yield per hectare

## 🌐 API Endpoints

- `GET /`: Display the prediction form
- `POST /predict`: Process form data and return yield prediction

## 📋 Data Format

The training dataset (`crop_yield.csv`) should contain the following columns:
- `Crop_Year`: Year (integer)
- `Area`: Area in hectares (float)
- `Production`: Production in tonnes (float)
- `Annual_Rainfall`: Rainfall in mm (float)
- `Fertilizer`: Fertilizer usage in kg/hectare (float)
- `Pesticide`: Pesticide usage in kg/hectare (float)
- `Crop`: Crop name (string)
- `Season`: Season name (string)
- `State`: State name (string)
- `Yield`: Target yield per hectare (float)

## 🔧 Configuration

### Server Configuration
- **Host**: 0.0.0.0 (accessible from all interfaces)
- **Port**: Configurable via PORT environment variable (default: 5000)
- **Debug Mode**: Enabled for development

### Environment Variables
- `PORT`: Server port number (optional, defaults to 5000)

## 📝 Example Usage

### Sample Input
```
Crop Year: 2023
Area: 100.5
Production: 500.2
Annual Rainfall: 1200.0
Fertilizer: 150.0
Pesticide: 25.0
Crop: Rice
Season: Kharif
State: Punjab
```

### Expected Output
```
Predicted Yield: 4.98 tonnes per hectare
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -am 'Add new feature'`)
6. Push to the branch (`git push origin feature/improvement`)
7. Create a Pull Request

## 🚀 Future Enhancements

- [ ] Add more sophisticated ML models (Random Forest, XGBoost)
- [ ] Implement model performance metrics and validation
- [ ] Add data visualization for trends and predictions
- [ ] Include weather API integration for real-time rainfall data
- [ ] Add user authentication and prediction history
- [ ] Implement batch prediction capabilities
- [ ] Add mobile-responsive design improvements
- [ ] Include crop recommendation system
- [ ] Add export functionality for predictions
- [ ] Implement A/B testing for different models

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**GSaiPhanindraPavanKumar**
- GitHub: [@GSaiPhanindraPavanKumar](https://github.com/GSaiPhanindraPavanKumar)

## 🙏 Acknowledgments

- scikit-learn community for the machine learning library
- Flask development team for the web framework
- Agricultural data providers for datasets
- Open source community for continuous support

## 📞 Support

If you encounter any issues or have questions:
1. Check the existing [Issues](https://github.com/GSaiPhanindraPavanKumar/Crop_Yield_Prediction/issues)
2. Create a new issue with detailed description
3. Provide steps to reproduce the problem

---

**Note**: This application is designed for educational and research purposes. For production agricultural decisions, please consult with agricultural experts and use multiple data sources.
