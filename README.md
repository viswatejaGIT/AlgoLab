# AlgoLab - ML Model Comparison Platform

A comprehensive machine learning model comparison tool built with Streamlit that allows users to compare 5 top ML algorithms across classification and regression tasks.

## Features

- **CSV/Excel Data Upload** - Support for multiple file formats
- **Dataset Statistical Info** - Comprehensive data analysis and visualization
- **5 Top ML Algorithm Comparison** - Compare the most popular algorithms
- **Performance Visualizations** - Interactive charts and metrics
- **Best Model Recommendations** - Automated model ranking and suggestions

## Supported Models

### Classification
- Logistic Regression
- Random Forest Classifier
- Support Vector Classifier
- Gradient Boosting Classifier
- K-Nearest Neighbors Classifier

### Regression
- Linear Regression
- Random Forest Regressor
- Support Vector Regressor
- Gradient Boosting Regressor
- Ridge Regressor

## Installation

1. Clone the repository:
```bash
git clone https://github.com/viswatejaGIT/AlgoLab.git
cd AlgoLab
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. **Upload Data**: Choose your CSV or Excel file
2. **Configure Model**: Select target variable and features
3. **Run Analysis**: Compare all 5 algorithms automatically
4. **View Results**: Interactive visualizations and performance metrics
5. **Get Recommendations**: See the top 2 performing models

## Technologies Used

- **Streamlit** - Web application framework
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning algorithms
- **Plotly** - Interactive visualizations
- **NumPy** - Numerical computing

## Project Structure

```
AlgoLab/
app.py              # Main application
requirements.txt    # Dependencies
README.md          # Project documentation
```
