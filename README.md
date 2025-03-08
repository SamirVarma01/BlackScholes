# Black-Scholes Option Pricing Calculator with ML

This application provides an interactive Black-Scholes option pricing calculator with machine learning capabilities.

## Features

- Black-Scholes option pricing for calls and puts
- Interactive controls for all key variables (S, K, T, r, σ)
- Visualization of option prices through heatmaps
- Machine learning model comparison
- What-if analysis for option pricing
- Feature importance visualization

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install streamlit numpy pandas matplotlib seaborn scipy scikit-learn joblib
```

## Files Structure

- `main.py` - Main Streamlit application
- `blackscholes.py` - Utility functions for Black-Scholes calculations
- `randomforest.py` - Machine learning model for option pricing
- `visualizations.py` - Functions for creating visualizations

## Usage

Run the application with:

```bash
streamlit run main.py
```

## Machine Learning Model

The application uses a Random Forest Regressor to predict option prices. The model:
- Trains on synthetically generated Black-Scholes prices
- Provides feature importance analysis
- Allows for inverse problem solving (finding parameter values for target prices)

The first time you run the application, it will train the ML models and save them for future use.
