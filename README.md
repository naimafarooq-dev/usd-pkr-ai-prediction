# USD/PKR Real-Time AI Prediction System

A real-time machine learning application that monitors the USD/PKR exchange rate, generates short-term predictions, compares Machine Learning predictions with a naive baseline, and evaluates prediction performance using real observed exchange-rate data.

The application is built with Python and Streamlit and uses a pre-trained online machine learning model to make continuous predictions from live market data.

## Project Overview

The **USD/PKR Real-Time AI Prediction System** is designed to demonstrate how machine learning can be integrated with real-time financial data.

The system:

* Fetches USD/PKR exchange-rate data from the Twelve Data API
* Collects historical market data for model input
* Generates real-time USD/PKR predictions
* Compares ML predictions with a naive baseline
* Calculates prediction errors
* Tracks prediction history
* Displays real-time charts and performance metrics
* Stores prediction results for later evaluation
* Provides an interactive Streamlit dashboard

## Key Features

### Real-Time Exchange Rate

The application retrieves current USD/PKR exchange-rate information through the Twelve Data API.

### Machine Learning Prediction

A Random Forest-based online learning model is used to generate short-term USD/PKR predictions.

The model uses engineered time-series features derived from the exchange-rate data.

### Baseline Comparison

The ML prediction is compared against a simple naive baseline.

This makes it possible to determine whether the machine learning model provides measurable improvement over a straightforward prediction strategy.

### Performance Evaluation

The application records:

* Actual exchange rate
* Predicted exchange rate
* Baseline prediction
* ML prediction error
* Baseline prediction error

The collected results can be used to calculate:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Hit Rate

### Interactive Dashboard

The Streamlit interface provides:

* Current USD/PKR rate
* Latest prediction
* Prediction error
* ML vs baseline comparison
* Prediction history
* Interactive charts
* Model performance metrics

## Technology Stack

| Technology      | Purpose                             |
| --------------- | ----------------------------------- |
| Python          | Core programming language           |
| Streamlit       | Interactive web dashboard           |
| Pandas          | Data processing                     |
| NumPy           | Numerical computations              |
| River           | Online machine learning             |
| Plotly          | Interactive visualizations          |
| Requests        | API requests                        |
| Joblib          | Model serialization                 |
| Twelve Data API | Real-time/historical USD/PKR data   |
| Git & GitHub    | Version control and project hosting |

## Machine Learning Approach

The project uses a **Random Forest-based machine learning model** together with feature engineering.

The system extracts information from the exchange-rate time series and uses these features to generate the next short-term prediction.

The prediction workflow is:

```text
Live USD/PKR Data
        ↓
Data Collection
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Machine Learning Model
        ↓
USD/PKR Prediction
        ↓
Actual Rate Arrives
        ↓
Prediction Error
        ↓
Model Evaluation
```

## Evaluation

The application records predictions and their corresponding actual values so that model performance can be evaluated using real observations.

For one collected evaluation period, the system achieved approximately:

| Metric             |   ML Model | Naive Baseline |
| ------------------ | ---------: | -------------: |
| MAE                | 0.0406 PKR |     0.0475 PKR |
| RMSE               | 0.0495 PKR |     0.0712 PKR |
| Hit Rate ±0.15 PKR |       100% |         90.32% |

For this evaluation sample, the ML model had approximately **14.5% lower MAE** than the baseline.

These results represent the specific collected evaluation sample and should not be interpreted as a guarantee of future market prediction performance.

## Project Structure

```text
usd-pkr-ai-prediction/
│
├── app.py
├── api.py
├── charts.py
├── config.py
├── data_manager.py
├── evaluate_model.py
├── features.py
├── metrics.py
├── model.py
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── prediction_history.csv
│   └── usd_pkr_data.csv
│
└── models/
    └── usd_pkr_model.pkl
```

## File Description

### `app.py`

Main Streamlit application.

It provides the dashboard interface and coordinates the data collection, prediction, evaluation, and visualization components.

### `api.py`

Handles communication with the Twelve Data API and retrieves USD/PKR exchange-rate data.

### `features.py`

Contains feature-engineering logic used to transform exchange-rate data into model features.

### `model.py`

Contains the machine learning model and model-loading/prediction functionality.

### `metrics.py`

Contains functions used to calculate model performance metrics such as MAE, RMSE, and hit rate.

### `data_manager.py`

Handles storage and management of collected exchange-rate and prediction data.

### `charts.py`

Creates interactive Plotly visualizations for the Streamlit dashboard.

### `evaluate_model.py`

Provides functionality for evaluating model predictions against actual exchange-rate observations.

### `config.py`

Contains project configuration such as:

* Data directories
* Model location
* API-related settings
* Display window
* Prediction thresholds
* Update interval

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/naimafarooq-dev/usd-pkr-ai-prediction.git
cd usd-pkr-ai-prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

On Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## API Configuration

The application uses the Twelve Data API.

The API key is stored through Streamlit secrets rather than directly inside the source code.

Create:

```text
.streamlit/secrets.toml
```

and add:

```toml
TWELVE_DATA_API_KEY = "YOUR_API_KEY"
```

Replace `YOUR_API_KEY` with your actual Twelve Data API key.

**Do not commit `secrets.toml` to GitHub.**

The project `.gitignore` already excludes:

```text
.streamlit/secrets.toml
```

## Running the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

The application will open in your browser and display the real-time USD/PKR prediction dashboard.

## Prediction Workflow

Each prediction follows this general process:

```text
1. Retrieve USD/PKR market data
            ↓
2. Prepare historical data
            ↓
3. Generate time-series features
            ↓
4. Generate ML prediction
            ↓
5. Display prediction
            ↓
6. Wait for actual exchange rate
            ↓
7. Calculate prediction error
            ↓
8. Compare with baseline
            ↓
9. Store result
```

## Data Storage

The project stores collected data locally.

### `data/usd_pkr_data.csv`

Contains historical USD/PKR exchange-rate data used by the application.

### `data/prediction_history.csv`

Stores prediction evaluation records including:

```text
prediction_time
actual_time
actual
predicted
baseline
ml_error
baseline_error
```

This allows the model's predictions to be evaluated against actual future observations.

## Security

API credentials are managed through Streamlit secrets.

Sensitive configuration files such as:

```text
.streamlit/secrets.toml
.env
.env.*
```

are excluded from version control.

The trained model file is included in the repository so that the deployed application can load the existing model without requiring local training.

## Limitations

This project is a machine learning demonstration and is not intended to provide financial advice or guaranteed exchange-rate forecasts.

Exchange rates are affected by many factors, including:

* Economic conditions
* Interest rates
* Inflation
* Monetary policy
* Political and geopolitical events
* Market liquidity
* Supply and demand

Short-term prediction performance can also change as market conditions change.

The reported evaluation metrics are based on the collected evaluation data available for this project and may not represent future performance.

## Future Improvements

Potential improvements include:

* Larger historical datasets
* Additional economic indicators
* More advanced time-series features
* Hyperparameter optimization
* Comparison with additional ML algorithms
* Deep learning models such as LSTM or GRU
* Automated model retraining
* Longer-term evaluation
* Model drift monitoring
* Cloud deployment
* Improved dashboard analytics
