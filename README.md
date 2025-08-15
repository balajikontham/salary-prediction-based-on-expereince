# Salary Prediction Model

A machine learning project that predicts salary based on years of experience using Linear Regression. The project includes both model training and a Streamlit web interface for predictions.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project demonstrates a simple yet effective salary prediction model using Linear Regression. It includes:
- Data analysis and visualization
- Machine learning model training
- Web interface for making predictions
- Statistical analysis of the dataset

## Features
- Predict salary based on years of experience
- Interactive web interface using Streamlit
- Data visualization of the salary vs experience relationship
- Statistical analysis of the dataset
- Pre-trained model included for immediate use

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd salary-prediction-based-on-experience
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Web App
To start the Streamlit web application:
```bash
streamlit run app.py
```

### Making Predictions
1. Enter the number of years of experience in the input field
2. Click the "Predict Salary" button
3. View the predicted salary

### Training the Model
If you want to retrain the model with new data:
1. Replace `Data.csv` with your dataset
2. Run the training script:
   ```bash
   python "Salary Prediction.py"
   ```
3. The script will generate a new `linear_regression_model.pkl` file

## Project Structure
```
salary-prediction/
├── Data.csv                  # Dataset containing experience and salary data
├── Salary Prediction.py      # Script for model training and analysis
├── app.py                    # Streamlit web application
└── linear_regression_model.pkl  # Pre-trained model
```

## Model Details
- **Algorithm**: Linear Regression
- **Input Feature**: Years of Experience
- **Target Variable**: Salary
- **Evaluation**: Model performance can be evaluated using the test set metrics in the training script

## Dependencies
- Python 3.6+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit
- SciPy

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is open source and available under the [MIT License](LICENSE).
