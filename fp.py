import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time  # For tracking forecast generation time

from prophet import Prophet
import pmdarima as pm
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)
from statsmodels.tsa.api import ExponentialSmoothing
from tbats import TBATS

from neuralprophet import NeuralProphet, configure

# Importing new libraries for machine learning models
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Set page configuration
st.set_page_config(page_title="Auto Forecasting App", layout="wide")

# App title and description
st.title('📈 Auto Forecasting App')
st.markdown("""
This app allows you to upload time series data and perform forecasting using multiple forecasting models.
""")

# Sidebar for file upload and settings
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        @st.cache_data
        def load_data(uploaded_file):
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                return data
            except Exception as e:
                st.error("Error reading file: " + str(e))
                st.stop()

        data = load_data(uploaded_file)

        # Select date column
        st.subheader("Select Date Column")
        date_column = st.selectbox("Date Column", data.columns, help="Select the column that contains date information.")
        data[date_column] = pd.to_datetime(data[date_column], errors='coerce')

        if data[date_column].isnull().any():
            st.error("Date column contains invalid dates.")
            st.stop()

        # Select data columns
        st.subheader("Select Data Columns to Forecast")
        data_columns = st.multiselect(
            "Data Columns",
            [col for col in data.columns if col != date_column],
            help="Select one or more columns you want to forecast.",
        )

        if not data_columns:
            st.error("Please select at least one data column to forecast.")
            st.stop()

        data.set_index(date_column, inplace=True)
        data.index.name = 'ds'  # Renaming index to 'ds' for consistency

        # Select exogenous variables (for models that support them)
        st.subheader("Select Exogenous Variables (Optional)")
        exogenous_columns = st.multiselect(
            "Exogenous Variables",
            [col for col in data.columns if col not in data_columns],
            help="Select columns to be used as additional regressors (optional).",
        )

        # Select forecasting library
        st.subheader("Select Forecasting Library")
        library_options = [
            'Prophet',
            'ARIMA/SARIMAX',
            'Exponential Smoothing',
            'TBATS',
            'NeuralProphet',
            'LightGBM',
            'XGBoost',
            'Random Forest'
        ]
        selected_library = st.selectbox(
            "Forecasting Library",
            library_options,
            help="Choose the forecasting library you want to use.",
        )

        # Based on selected library, select models and parameters
        # Using st.expander to hide advanced settings
        if selected_library == 'NeuralProphet':
            with st.expander("NeuralProphet Model Parameters"):
                np_params = {}
                np_params['seasonality_mode'] = st.selectbox(
                    "Seasonality Mode",
                    ['additive', 'multiplicative'],
                    help="Type of seasonality model.",
                )
                np_params['n_forecasts'] = st.number_input(
                    "Number of Forecast Steps",
                    min_value=1,
                    value=1,
                    help="Number of future steps to forecast.",
                )
                np_params['n_lags'] = st.number_input(
                    "Number of Lags",
                    min_value=0,
                    value=0,
                    help="Number of past observations to use.",
                )
                np_params['num_hidden_layers'] = st.number_input(
                    "Number of Hidden Layers",
                    min_value=0,
                    value=0,
                    help="Number of hidden layers in the neural network.",
                )
                np_params['d_hidden'] = st.number_input(
                    "Hidden Layer Dimension",
                    min_value=0,
                    value=0,
                    help="Dimension of each hidden layer.",
                )
                np_params['dropout'] = st.slider(
                    "Dropout Rate",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    help="Dropout rate for regularization.",
                )
                np_params['learning_rate'] = st.number_input(
                    "Learning Rate",
                    min_value=0.0001,
                    value=0.001,
                    format="%.4f",
                    help="Learning rate for optimization.",
                )
                np_params['epochs'] = st.number_input(
                    "Number of Epochs",
                    min_value=1,
                    value=100,
                    help="Total number of training epochs.",
                )
        elif selected_library == 'LightGBM':
            with st.expander("LightGBM Parameters"):
                lgbm_params = {}
                lgbm_params['num_leaves'] = st.number_input(
                    "Number of Leaves",
                    min_value=2,
                    value=31,
                    help="Number of leaves in one tree.",
                )
                lgbm_params['max_depth'] = st.number_input(
                    "Max Depth",
                    min_value=-1,
                    value=-1,
                    help="Maximum tree depth for base learners, <=0 means no limit.",
                )
                lgbm_params['learning_rate'] = st.number_input(
                    "Learning Rate",
                    min_value=0.0001,
                    value=0.1,
                    format="%.4f",
                    help="Boosting learning rate.",
                )
                lgbm_params['n_estimators'] = st.number_input(
                    "Number of Estimators",
                    min_value=1,
                    value=100,
                    help="Number of boosting iterations.",
                )
                lgbm_params['min_child_samples'] = st.number_input(
                    "Minimum Child Samples",
                    min_value=1,
                    value=20,
                    help="Minimum number of data needed in a child (leaf).",
                )
                lgbm_params['subsample'] = st.slider(
                    "Subsample (Bagging Fraction)",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0,
                    help="Subsample ratio of the training instance.",
                )
                lgbm_params['colsample_bytree'] = st.slider(
                    "Colsample Bytree (Feature Fraction)",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0,
                    help="Subsample ratio of columns when constructing each tree.",
                )
                # Number of lags to create
                lgbm_params['n_lags'] = st.number_input(
                    "Number of Lags",
                    min_value=1,
                    value=5,
                    help="Number of lag features to create.",
                )
        elif selected_library == 'XGBoost':
            with st.expander("XGBoost Parameters"):
                xgb_params = {}
                xgb_params['max_depth'] = st.number_input(
                    "Max Depth",
                    min_value=1,
                    value=6,
                    help="Maximum tree depth for base learners.",
                )
                xgb_params['learning_rate'] = st.number_input(
                    "Learning Rate",
                    min_value=0.0001,
                    value=0.3,
                    format="%.4f",
                    help="Boosting learning rate.",
                )
                xgb_params['n_estimators'] = st.number_input(
                    "Number of Estimators",
                    min_value=1,
                    value=100,
                    help="Number of boosting rounds.",
                )
                xgb_params['subsample'] = st.slider(
                    "Subsample",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0,
                    help="Subsample ratio of the training instance.",
                )
                xgb_params['colsample_bytree'] = st.slider(
                    "Colsample Bytree",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0,
                    help="Subsample ratio of columns when constructing each tree.",
                )
                xgb_params['gamma'] = st.number_input(
                    "Gamma",
                    min_value=0.0,
                    value=0.0,
                    help="Minimum loss reduction required to make a further partition on a leaf node of the tree.",
                )
                xgb_params['reg_alpha'] = st.number_input(
                    "Alpha (L1 Regularization)",
                    min_value=0.0,
                    value=0.0,
                    help="L1 regularization term on weights.",
                )
                xgb_params['reg_lambda'] = st.number_input(
                    "Lambda (L2 Regularization)",
                    min_value=0.0,
                    value=1.0,
                    help="L2 regularization term on weights.",
                )
                xgb_params['n_lags'] = st.number_input(
                    "Number of Lags",
                    min_value=1,
                    value=5,
                    help="Number of lag features to create.",
                )
        elif selected_library == 'Random Forest':
            with st.expander("Random Forest Parameters"):
                rf_params = {}
                rf_params['n_estimators'] = st.number_input(
                    "Number of Estimators",
                    min_value=1,
                    value=100,
                    help="Number of trees in the forest.",
                )
                rf_params['max_depth'] = st.number_input(
                    "Max Depth",
                    min_value=1,
                    value=None,
                    help="Maximum depth of the tree.",
                )
                rf_params['min_samples_split'] = st.number_input(
                    "Minimum Samples Split",
                    min_value=2,
                    value=2,
                    help="Minimum number of samples required to split an internal node.",
                )
                rf_params['min_samples_leaf'] = st.number_input(
                    "Minimum Samples Leaf",
                    min_value=1,
                    value=1,
                    help="Minimum number of samples required to be at a leaf node.",
                )
                rf_params['max_features'] = st.selectbox(
                    "Max Features",
                    options=['auto', 'sqrt', 'log2', None],
                    index=0,
                    help="The number of features to consider when looking for the best split.",
                )
                rf_params['bootstrap'] = st.selectbox(
                    "Bootstrap Samples",
                    options=[True, False],
                    index=0,
                    help="Whether bootstrap samples are used when building trees.",
                )
                rf_params['n_lags'] = st.number_input(
                    "Number of Lags",
                    min_value=1,
                    value=5,
                    help="Number of lag features to create.",
                )
        else:
            # Existing models from previous code
            if selected_library == 'Prophet':
                with st.expander("Prophet Model Parameters"):
                    prophet_params = {}
                    prophet_params['growth'] = st.selectbox(
                        "Growth",
                        ['linear', 'logistic'],
                        index=0,
                        help="Type of trend growth: 'linear' or 'logistic'.",
                    )
                    if prophet_params['growth'] == 'logistic':
                        # Need to set 'cap' and optionally 'floor' in the data
                        st.info("Since you selected 'logistic' growth, you need to specify 'cap' and optionally 'floor' in your data.")
                        cap_value = st.number_input(
                            "Cap (capacity) value",
                            value=1.0,
                            help="Maximum capacity for logistic growth.",
                        )
                        floor_value = st.number_input(
                            "Floor value",
                            value=0.0,
                            help="Minimum value for logistic growth.",
                        )
                    else:
                        cap_value = None
                        floor_value = None
                    prophet_params['changepoints'] = st.text_input(
                        "Changepoints (comma-separated dates, YYYY-MM-DD)",
                        value="",
                        help="Specific dates at which to include potential changepoints.",
                    )
                    prophet_params['n_changepoints'] = st.number_input(
                        "Number of Changepoints",
                        min_value=0,
                        value=25,
                        help="Number of potential changepoints to include.",
                    )
                    prophet_params['changepoint_range'] = st.slider(
                        "Changepoint Range",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.8,
                        help="Proportion of history in which trend changepoints will be estimated.",
                    )
                    prophet_params['changepoint_prior_scale'] = st.number_input(
                        "Changepoint Prior Scale",
                        min_value=0.001,
                        value=0.05,
                        format="%.3f",
                        help="Parameter controlling flexibility of trend changepoints.",
                    )
                    prophet_params['seasonality_prior_scale'] = st.number_input(
                        "Seasonality Prior Scale",
                        min_value=0.01,
                        value=10.0,
                        format="%.2f",
                        help="Parameter controlling flexibility of seasonality.",
                    )
                    prophet_params['holidays_prior_scale'] = st.number_input(
                        "Holidays Prior Scale",
                        min_value=0.01,
                        value=10.0,
                        format="%.2f",
                        help="Parameter controlling flexibility of holiday effects.",
                    )
                    prophet_params['seasonality_mode'] = st.selectbox(
                        "Seasonality Mode",
                        ['additive', 'multiplicative'],
                        index=0,
                        help="Mode for seasonality: 'additive' or 'multiplicative'.",
                    )
                    prophet_params['yearly_seasonality'] = st.radio(
                        "Yearly Seasonality",
                        options=['auto', 'True', 'False', 'Custom'],
                        index=0,
                        help="Fit yearly seasonality: 'auto', True, False, or 'Custom' for custom order.",
                    )
                    if prophet_params['yearly_seasonality'] == 'Custom':
                        prophet_params['yearly_seasonality'] = st.number_input(
                            "Yearly Seasonality Fourier Order",
                            min_value=1,
                            value=10,
                            help="Fourier order for yearly seasonality.",
                        )
                    elif prophet_params['yearly_seasonality'] == 'True':
                        prophet_params['yearly_seasonality'] = True
                    elif prophet_params['yearly_seasonality'] == 'False':
                        prophet_params['yearly_seasonality'] = False
                    else:
                        prophet_params['yearly_seasonality'] = 'auto'

                    prophet_params['weekly_seasonality'] = st.radio(
                        "Weekly Seasonality",
                        options=['auto', 'True', 'False', 'Custom'],
                        index=0,
                        help="Fit weekly seasonality: 'auto', True, False, or 'Custom' for custom order.",
                    )
                    if prophet_params['weekly_seasonality'] == 'Custom':
                        prophet_params['weekly_seasonality'] = st.number_input(
                            "Weekly Seasonality Fourier Order",
                            min_value=1,
                            value=3,
                            help="Fourier order for weekly seasonality.",
                        )
                    elif prophet_params['weekly_seasonality'] == 'True':
                        prophet_params['weekly_seasonality'] = True
                    elif prophet_params['weekly_seasonality'] == 'False':
                        prophet_params['weekly_seasonality'] = False
                    else:
                        prophet_params['weekly_seasonality'] = 'auto'

                    prophet_params['daily_seasonality'] = st.radio(
                        "Daily Seasonality",
                        options=['auto', 'True', 'False', 'Custom'],
                        index=0,
                        help="Fit daily seasonality: 'auto', True, False, or 'Custom' for custom order.",
                    )
                    if prophet_params['daily_seasonality'] == 'Custom':
                        prophet_params['daily_seasonality'] = st.number_input(
                            "Daily Seasonality Fourier Order",
                            min_value=1,
                            value=4,
                            help="Fourier order for daily seasonality.",
                        )
                    elif prophet_params['daily_seasonality'] == 'True':
                        prophet_params['daily_seasonality'] = True
                    elif prophet_params['daily_seasonality'] == 'False':
                        prophet_params['daily_seasonality'] = False
                    else:
                        prophet_params['daily_seasonality'] = 'auto'

                    # Additional seasonality terms
                    st.subheader("Additional Seasonality Terms")
                    custom_seasonality = st.text_input(
                        "Custom Seasonalities (name;period;fourier_order - separate multiple by commas)",
                        value="",
                        help="Add custom seasonalities in the format: name;period;fourier_order.",
                    )
                    prophet_params['interval_width'] = st.slider(
                        "Interval Width",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.80,
                        help="Width of the uncertainty intervals provided for the forecast.",
                    )
                    prophet_params['mcmc_samples'] = st.number_input(
                        "MCMC Samples",
                        min_value=0,
                        value=0,
                        help="Number of MCMC samples for full Bayesian inference.",
                    )
                    prophet_params['uncertainty_samples'] = st.number_input(
                        "Uncertainty Samples",
                        min_value=0,
                        value=1000,
                        help="Number of simulated draws used to estimate uncertainty intervals.",
                    )
                    # Holidays
                    st.subheader("Holidays")
                    use_holidays = st.checkbox(
                        "Use Holidays",
                        value=False,
                        help="Include holiday effects in the model.",
                    )
                    if use_holidays:
                        # The user can upload a DataFrame with holidays
                        st.info("Upload a CSV or Excel file with holidays. It should have columns 'ds' (date), 'holiday' (name), and optionally 'lower_window', 'upper_window', 'prior_scale'")
                        holidays_file = st.file_uploader(
                            "Upload Holidays File",
                            type=["csv", "xlsx"],
                            help="Upload a file containing holiday dates and names.",
                        )
                        if holidays_file is not None:
                            try:
                                if holidays_file.name.endswith('.csv'):
                                    holidays = pd.read_csv(holidays_file)
                                else:
                                    holidays = pd.read_excel(holidays_file)
                                # Ensure 'ds' column is datetime
                                holidays['ds'] = pd.to_datetime(holidays['ds'], errors='coerce')
                                if holidays['ds'].isnull().any():
                                    st.error("Holidays 'ds' column contains invalid dates.")
                                    holidays = None
                            except Exception as e:
                                st.error("Error reading holidays file: " + str(e))
                                holidays = None
                        else:
                            holidays = None
                    else:
                        holidays = None
                    # Custom Seasonalities
                    st.subheader("Custom Seasonalities")
                    add_seasonality = st.checkbox(
                        "Add Custom Seasonality",
                        value=False,
                        help="Option to add a custom seasonality component.",
                    )
                    if add_seasonality:
                        seasonality_name = st.text_input(
                            "Seasonality Name",
                            value="monthly",
                            help="Name of the custom seasonality.",
                        )
                        period = st.number_input(
                            "Period",
                            min_value=0.0,
                            value=30.5,
                            help="Period of the seasonality in days.",
                        )
                        fourier_order = st.number_input(
                            "Fourier Order",
                            min_value=1,
                            value=5,
                            help="Number of Fourier components to include.",
                        )
                        prior_scale = st.number_input(
                            "Prior Scale",
                            min_value=0.01,
                            value=10.0,
                            help="Strength of the seasonality model.",
                        )
                    else:
                        seasonality_name = None

                    # Cross-Validation Settings
                    st.subheader("Cross-Validation Settings")
                    perform_cross_validation = st.checkbox("Perform Cross-Validation", value=False, help="Check to perform cross-validation for model evaluation.")
                    if perform_cross_validation:
                        initial = st.text_input("Initial Training Period (e.g., '365 days')", value='730 days', help="Initial training period length.")
                        period = st.text_input("Period Between Cutoffs (e.g., '180 days')", value='180 days', help="Spacing between cutoff dates.")
                        horizon = st.text_input("Forecast Horizon (e.g., '365 days')", value='365 days', help="Forecast horizon for cross-validation.")

            elif selected_library == 'ARIMA/SARIMAX':
                with st.expander("ARIMA/SARIMAX Model Parameters"):
                    arima_params = {}
                    arima_params['start_p'] = st.number_input(
                        "Start p",
                        min_value=0,
                        max_value=10,
                        value=1,
                        help="Starting value of p in the grid search.",
                    )
                    arima_params['start_q'] = st.number_input(
                        "Start q",
                        min_value=0,
                        max_value=10,
                        value=1,
                        help="Starting value of q in the grid search.",
                    )
                    arima_params['max_p'] = st.number_input(
                        "Max p",
                        min_value=0,
                        max_value=10,
                        value=3,
                        help="Maximum value of p in the grid search.",
                    )
                    arima_params['max_q'] = st.number_input(
                        "Max q",
                        min_value=0,
                        max_value=10,
                        value=3,
                        help="Maximum value of q in the grid search.",
                    )
                    arima_params['m'] = st.number_input(
                        "Seasonal Period (m)",
                        min_value=1,
                        value=1,
                        help="Number of periods in each season.",
                    )
                    arima_params['seasonal'] = st.checkbox(
                        "Seasonal",
                        value=False,
                        help="Whether to fit a seasonal ARIMA.",
                    )
                    arima_params['start_P'] = st.number_input(
                        "Start P (Seasonal)",
                        min_value=0,
                        max_value=10,
                        value=0,
                        help="Starting value of P in the seasonal grid search.",
                    )
                    arima_params['start_Q'] = st.number_input(
                        "Start Q (Seasonal)",
                        min_value=0,
                        max_value=10,
                        value=0,
                        help="Starting value of Q in the seasonal grid search.",
                    )
                    arima_params['max_P'] = st.number_input(
                        "Max P (Seasonal)",
                        min_value=0,
                        max_value=10,
                        value=1,
                        help="Maximum value of P in the seasonal grid search.",
                    )
                    arima_params['max_Q'] = st.number_input(
                        "Max Q (Seasonal)",
                        min_value=0,
                        max_value=10,
                        value=1,
                        help="Maximum value of Q in the seasonal grid search.",
                    )
                    arima_params['d'] = st.number_input(
                        "Order of Differencing (d)",
                        min_value=0,
                        max_value=10,
                        value=None,
                        help="Order of differencing for non-seasonal data.",
                    )
                    arima_params['D'] = st.number_input(
                        "Order of Seasonal Differencing (D)",
                        min_value=0,
                        max_value=10,
                        value=None,
                        help="Order of differencing for seasonal data.",
                    )
                    arima_params['suppress_warnings'] = True
                    arima_params['error_action'] = 'ignore'
                    arima_params['stepwise'] = st.checkbox(
                        "Use Stepwise Selection",
                        value=True,
                        help="Whether to use stepwise algorithm for order selection.",
                    )
                    arima_params['n_fits'] = st.number_input(
                        "Number of Fits (if stepwise=False)",
                        min_value=1,
                        value=10,
                        help="Number of ARIMA models to fit if stepwise is False.",
                    )
            elif selected_library == 'Exponential Smoothing':
                with st.expander("Exponential Smoothing Parameters"):
                    es_params = {}
                    es_params['trend'] = st.selectbox(
                        "Trend Component",
                        [None, 'add', 'mul'],
                        help="Type of trend component to include.",
                    )
                    es_params['damped_trend'] = st.checkbox(
                        "Damped Trend",
                        value=False,
                        help="Whether to include a damped trend component.",
                    )
                    es_params['seasonal'] = st.selectbox(
                        "Seasonal Component",
                        [None, 'add', 'mul'],
                        help="Type of seasonal component to include.",
                    )
                    es_params['seasonal_periods'] = st.number_input(
                        "Seasonal Periods",
                        min_value=1,
                        value=12,
                        help="Number of periods in a complete seasonal cycle.",
                    )
                    es_params['use_boxcox'] = st.selectbox(
                        "Use Box-Cox Transformation",
                        [True, False, 'log'],
                        index=1,
                        help="Apply Box-Cox transformation to the data.",
                    )
                    es_params['initialization_method'] = st.selectbox(
                        "Initialization Method",
                        ['estimated', 'heuristic', 'legacy-heuristic', 'known'],
                        index=0,
                        help="Method for initializing the model parameters.",
                    )
            elif selected_library == 'TBATS':
                with st.expander("TBATS Model Parameters"):
                    tbats_params = {}
                    tbats_params['use_box_cox'] = st.checkbox(
                        "Use Box-Cox Transformation",
                        value=False,
                        help="Apply Box-Cox transformation to stabilize variance.",
                    )
                    tbats_params['use_trend'] = st.checkbox(
                        "Use Trend",
                        value=True,
                        help="Include a trend component in the model.",
                    )
                    tbats_params['use_damped_trend'] = st.checkbox(
                        "Use Damped Trend",
                        value=False,
                        help="Include a damped trend component.",
                    )
                    tbats_params['sp_list'] = st.text_input(
                        "Seasonal Periods (comma-separated)",
                        value="12",
                        help="List of seasonal periods to model.",
                    )
                    # Parse seasonal periods
                    sp_list = [int(s.strip()) for s in tbats_params['sp_list'].split(',') if s.strip().isdigit()]
                    tbats_params['seasonal_periods'] = sp_list

        # Select forecast frequency
        st.subheader("Forecast Settings")
        st.write("Select Forecast Frequency for the output:")
        forecast_frequency = st.selectbox(
            "Forecast Frequency",
            ['Daily', 'Weekly', 'Monthly'],
            help="Frequency at which to generate forecasts.",
        )

        if forecast_frequency == 'Daily':
            freq_str = 'D'
        elif forecast_frequency == 'Weekly':
            freq_str = 'W'
        elif forecast_frequency == 'Monthly':
            freq_str = 'M'

        # Forecast horizon
        st.write("Select Forecast Horizon:")
        periods_input = st.number_input(
            'How many periods would you like to forecast into the future?',
            min_value=1,
            value=30,
            help="Number of periods to forecast ahead.",
        )

        # Add the "Generate Forecast" button
        generate_forecast = st.button("Generate Forecast")

if uploaded_file is not None and data_columns and 'generate_forecast' in locals() and generate_forecast:
    forecasts = {}
    tabs = st.tabs(data_columns)
    for i, col in enumerate(data_columns):
        with tabs[i]:
            st.header(f"Forecasting for **{col}**")
            df = data[[col]].copy()

            # Resample data if forecast frequency is different from data frequency
            df = df.asfreq(freq_str)
            df = df.fillna(method='ffill')  # Fill missing values

            # Prepare exogenous variables if any
            if exogenous_columns:
                exog = data[exogenous_columns].copy()
                exog = exog.asfreq(freq_str)
                exog = exog.fillna(method='ffill')
            else:
                exog = None

            # Customization options
            st.subheader("Plot Customization")
            with st.expander("Customize Plot Appearance"):
                col1, col2 = st.columns(2)
                with col1:
                    line_color_actual = st.color_picker("Actual Data Line Color", "#1f77b4", key=f"actual_color_{col}")
                    line_style_actual = st.selectbox("Actual Data Line Style", ["solid", "dash", "dot", "dashdot"], key=f"actual_style_{col}")
                with col2:
                    line_color_forecast = st.color_picker("Forecast Line Color", "#ff7f0e", key=f"forecast_color_{col}")
                    line_style_forecast = st.selectbox("Forecast Line Style", ["solid", "dash", "dot", "dashdot"], key=f"forecast_style_{col}")
                include_confidence_interval = st.checkbox("Include Confidence Intervals (if available)", value=True, key=f"conf_interval_{col}")

            # Start timing the forecast generation
            start_time = time.time()

            if selected_library == 'Prophet':
                # Prepare data for Prophet
                df_prophet = df.reset_index().rename(columns={'ds': 'ds', col: 'y'})

                # Handle logistic growth
                if prophet_params['growth'] == 'logistic':
                    df_prophet['cap'] = cap_value
                    df_prophet['floor'] = floor_value

                # Parse changepoints
                if prophet_params['changepoints']:
                    changepoints_list = [s.strip() for s in prophet_params['changepoints'].split(',')]
                    changepoints_list = pd.to_datetime(changepoints_list)
                else:
                    changepoints_list = None

                # Initialize Prophet model with parameters
                model = Prophet(
                    growth=prophet_params['growth'],
                    changepoints=changepoints_list,
                    n_changepoints=prophet_params['n_changepoints'],
                    changepoint_range=prophet_params['changepoint_range'],
                    changepoint_prior_scale=prophet_params['changepoint_prior_scale'],
                    seasonality_prior_scale=prophet_params['seasonality_prior_scale'],
                    holidays_prior_scale=prophet_params['holidays_prior_scale'],
                    seasonality_mode=prophet_params['seasonality_mode'],
                    yearly_seasonality=prophet_params['yearly_seasonality'],
                    weekly_seasonality=prophet_params['weekly_seasonality'],
                    daily_seasonality=prophet_params['daily_seasonality'],
                    interval_width=prophet_params['interval_width'],
                    mcmc_samples=int(prophet_params['mcmc_samples']),
                    uncertainty_samples=int(prophet_params['uncertainty_samples']),
                    holidays=holidays
                )

                # Add additional custom seasonalities
                if custom_seasonality:
                    seasonality_terms = [s.strip() for s in custom_seasonality.split(',')]
                    for term in seasonality_terms:
                        parts = term.split(';')
                        if len(parts) == 3:
                            name, period, fourier_order = parts
                            name = name.strip()
                            period = float(period.strip())
                            fourier_order = int(fourier_order.strip())
                            model.add_seasonality(name=name, period=period, fourier_order=fourier_order)
                        else:
                            st.warning(f"Invalid custom seasonality term: {term}")

                if add_seasonality and seasonality_name:
                    model.add_seasonality(name=seasonality_name, period=period, fourier_order=fourier_order, prior_scale=prior_scale)

                # Add regressors if exogenous variables are provided
                if exog is not None:
                    for ex_col in exogenous_columns:
                        model.add_regressor(ex_col)
                    df_prophet = df_prophet.merge(exog.reset_index(), on='ds')

                with st.spinner("Training the Prophet model..."):
                    model.fit(df_prophet)

                # For logistic growth, need to specify 'cap' and 'floor' in future dataframe
                future = model.make_future_dataframe(periods=periods_input, freq=freq_str)

                if prophet_params['growth'] == 'logistic':
                    future['cap'] = cap_value
                    future['floor'] = floor_value

                if exog is not None:
                    future = future.merge(exog.reset_index(), on='ds', how='left')
                    future.fillna(method='ffill', inplace=True)

                forecast = model.predict(future)
                forecasts[col] = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].set_index('ds')

                # Model explainability: Prophet component plots
                st.subheader("Model Explainability")

                with st.spinner("Generating component plots..."):
                    fig_components = model.plot_components(forecast)
                    st.write(fig_components)

                # Cross-validation
                if perform_cross_validation:
                    with st.spinner("Performing cross-validation..."):
                        from prophet.diagnostics import cross_validation, performance_metrics
                        df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon)
                        df_p = performance_metrics(df_cv)
                        st.subheader("Cross-Validation Performance Metrics")
                        st.write(df_p)

            elif selected_library == 'ARIMA/SARIMAX':
                y = df[col]
                try:
                    if exog is not None:
                        exog_train = exog.loc[y.index]
                    else:
                        exog_train = None

                    with st.spinner("Training the ARIMA/SARIMAX model..."):
                        model = pm.auto_arima(
                            y,
                            exogenous=exog_train,
                            start_p=arima_params['start_p'],
                            start_q=arima_params['start_q'],
                            max_p=arima_params['max_p'],
                            max_q=arima_params['max_q'],
                            d=arima_params['d'],
                            start_P=arima_params['start_P'],
                            start_Q=arima_params['start_Q'],
                            max_P=arima_params['max_P'],
                            max_Q=arima_params['max_Q'],
                            D=arima_params['D'],
                            m=arima_params['m'],
                            seasonal=arima_params['seasonal'],
                            error_action=arima_params['error_action'],
                            suppress_warnings=arima_params['suppress_warnings'],
                            stepwise=arima_params['stepwise'],
                            n_fits=arima_params['n_fits']
                        )
                    n_periods = periods_input
                    index_of_fc = pd.date_range(y.index[-1], periods=n_periods+1, freq=freq_str)
                    index_of_fc = index_of_fc[1:]  # remove start date

                    if exog is not None:
                        exog_future = exog.reindex(index_of_fc)
                        exog_future.fillna(method='ffill', inplace=True)
                    else:
                        exog_future = None

                    forecast, conf_int = model.predict(n_periods=n_periods, exogenous=exog_future, return_conf_int=True, alpha=0.05)
                    fc_series = pd.Series(forecast, index=index_of_fc)
                    forecasts[col] = fc_series.to_frame(name='Forecast')
                    conf_int_df = pd.DataFrame(conf_int, index=index_of_fc, columns=['lower', 'upper'])
                except Exception as e:
                    st.error(f"ARIMA/SARIMAX model failed for {col}: {e}")
                    continue
            elif selected_library == 'Exponential Smoothing':
                y = df[col]
                try:
                    with st.spinner("Training the Exponential Smoothing model..."):
                        model = ExponentialSmoothing(
                            y,
                            trend=es_params['trend'],
                            damped_trend=es_params['damped_trend'],
                            seasonal=es_params['seasonal'],
                            seasonal_periods=es_params['seasonal_periods'],
                            initialization_method=es_params['initialization_method']
                        )
                        model_fit = model.fit(use_boxcox=es_params['use_boxcox'])
                    forecast = model_fit.forecast(periods_input)
                    index_of_fc = pd.date_range(y.index[-1], periods=periods_input+1, freq=freq_str)
                    index_of_fc = index_of_fc[1:]  # remove start date
                    forecasts[col] = pd.Series(forecast, index=index_of_fc).to_frame(name='Forecast')
                except Exception as e:
                    st.error(f"Exponential Smoothing model failed for {col}: {e}")
                    continue
            elif selected_library == 'TBATS':
                y = df[col]
                try:
                    with st.spinner("Training the TBATS model..."):
                        estimator = TBATS(
                            seasonal_periods=tbats_params['seasonal_periods'],
                            use_box_cox=tbats_params['use_box_cox'],
                            use_trend=tbats_params['use_trend'],
                            use_damped_trend=tbats_params['use_damped_trend']
                        )
                        model = estimator.fit(y)
                    forecast = model.forecast(steps=periods_input)
                    index_of_fc = pd.date_range(y.index[-1], periods=periods_input+1, freq=freq_str)
                    index_of_fc = index_of_fc[1:]  # remove start date
                    fc_series = pd.Series(forecast, index=index_of_fc)
                    forecasts[col] = fc_series.to_frame(name='Forecast')
                except Exception as e:
                    st.error(f"TBATS model failed for {col}: {e}")
                    continue
            elif selected_library == 'NeuralProphet':
                try:
                    df_np = df.reset_index().rename(columns={'ds': 'ds', col: 'y'})
                    if exog is not None:
                        df_exog = exog.reset_index()
                        df_np = df_np.merge(df_exog, on='ds', how='left')
                    ar_net = configure.ArNet(
                        num_hidden_layers=int(np_params['num_hidden_layers']),
                        d_hidden=int(np_params['d_hidden']),
                        dropout=np_params['dropout'],
                    )
                    with st.spinner("Training the NeuralProphet model..."):
                        model = NeuralProphet(
                            n_forecasts=int(np_params['n_forecasts']),
                            n_lags=int(np_params['n_lags']),
                            ar_net=ar_net,
                            seasonality_mode=np_params['seasonality_mode'],
                            learning_rate=np_params['learning_rate'],
                            epochs=int(np_params['epochs'])
                        )
                        if exog is not None:
                            for ex_col in exogenous_columns:
                                model.add_future_regressor(ex_col)
                        metrics = model.fit(df_np, freq=freq_str)
                    future = model.make_future_dataframe(df_np, periods=periods_input, n_historic_predictions=len(df_np))
                    if exog is not None:
                        future_exog = exog.reset_index()
                        future = future.merge(future_exog, on='ds', how='left')
                        future.fillna(method='ffill', inplace=True)
                    forecast = model.predict(future)
                    forecast.set_index('ds', inplace=True)
                    forecasts[col] = forecast['yhat1'].to_frame(name='Forecast')
                except Exception as e:
                    st.error(f"NeuralProphet model failed for {col}: {e}")
                    continue
            elif selected_library == 'LightGBM':
                y = df[col]
                try:
                    with st.spinner("Training the LightGBM model..."):
                        # Create lag features
                        df_lgbm = df.copy()
                        for lag in range(1, lgbm_params['n_lags'] + 1):
                            df_lgbm[f'lag_{lag}'] = df_lgbm[col].shift(lag)
                        # Add exogenous variables if any
                        if exog is not None:
                            df_lgbm = df_lgbm.merge(exog, left_index=True, right_index=True)
                        df_lgbm.dropna(inplace=True)
                        X = df_lgbm.drop(columns=[col])
                        y = df_lgbm[col]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=periods_input)
                        model = LGBMRegressor(
                            num_leaves=int(lgbm_params['num_leaves']),
                            max_depth=int(lgbm_params['max_depth']),
                            learning_rate=lgbm_params['learning_rate'],
                            n_estimators=int(lgbm_params['n_estimators']),
                            min_child_samples=int(lgbm_params['min_child_samples']),
                            subsample=lgbm_params['subsample'],
                            colsample_bytree=lgbm_params['colsample_bytree']
                        )
                        model.fit(X_train, y_train)
                        # Forecast
                        last_values = X_train.iloc[-1].values.reshape(1, -1)
                        forecasts_list = []
                        for _ in range(periods_input):
                            y_pred = model.predict(last_values)
                            forecasts_list.append(y_pred[0])
                            # Update last_values with new prediction
                            last_values = np.roll(last_values, -1)
                            last_values[0, -1] = y_pred[0]
                    index_of_fc = pd.date_range(df_lgbm.index[-1], periods=periods_input+1, freq=freq_str)[1:]
                    forecasts[col] = pd.Series(forecasts_list, index=index_of_fc).to_frame(name='Forecast')
                except Exception as e:
                    st.error(f"LightGBM model failed for {col}: {e}")
                    continue
            elif selected_library == 'XGBoost':
                y = df[col]
                try:
                    with st.spinner("Training the XGBoost model..."):
                        # Create lag features
                        df_xgb = df.copy()
                        for lag in range(1, xgb_params['n_lags'] + 1):
                            df_xgb[f'lag_{lag}'] = df_xgb[col].shift(lag)
                        # Add exogenous variables if any
                        if exog is not None:
                            df_xgb = df_xgb.merge(exog, left_index=True, right_index=True)
                        df_xgb.dropna(inplace=True)
                        X = df_xgb.drop(columns=[col])
                        y = df_xgb[col]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=periods_input)
                        model = xgb.XGBRegressor(
                            max_depth=int(xgb_params['max_depth']),
                            learning_rate=xgb_params['learning_rate'],
                            n_estimators=int(xgb_params['n_estimators']),
                            subsample=xgb_params['subsample'],
                            colsample_bytree=xgb_params['colsample_bytree'],
                            gamma=xgb_params['gamma'],
                            reg_alpha=xgb_params['reg_alpha'],
                            reg_lambda=xgb_params['reg_lambda']
                        )
                        model.fit(X_train, y_train)
                        # Forecast
                        last_values = X_train.iloc[-1].values.reshape(1, -1)
                        forecasts_list = []
                        for _ in range(periods_input):
                            y_pred = model.predict(last_values)
                            forecasts_list.append(y_pred[0])
                            # Update last_values with new prediction
                            last_values = np.roll(last_values, -1)
                            last_values[0, -1] = y_pred[0]
                    index_of_fc = pd.date_range(df_xgb.index[-1], periods=periods_input+1, freq=freq_str)[1:]
                    forecasts[col] = pd.Series(forecasts_list, index=index_of_fc).to_frame(name='Forecast')
                except Exception as e:
                    st.error(f"XGBoost model failed for {col}: {e}")
                    continue
            elif selected_library == 'Random Forest':
                y = df[col]
                try:
                    with st.spinner("Training the Random Forest model..."):
                        # Create lag features
                        df_rf = df.copy()
                        for lag in range(1, rf_params['n_lags'] + 1):
                            df_rf[f'lag_{lag}'] = df_rf[col].shift(lag)
                        # Add exogenous variables if any
                        if exog is not None:
                            df_rf = df_rf.merge(exog, left_index=True, right_index=True)
                        df_rf.dropna(inplace=True)
                        X = df_rf.drop(columns=[col])
                        y = df_rf[col]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=periods_input)
                        model = RandomForestRegressor(
                            n_estimators=int(rf_params['n_estimators']),
                            max_depth=rf_params['max_depth'] if rf_params['max_depth'] != 1 else None,
                            min_samples_split=int(rf_params['min_samples_split']),
                            min_samples_leaf=int(rf_params['min_samples_leaf']),
                            max_features=rf_params['max_features'],
                            bootstrap=rf_params['bootstrap']
                        )
                        model.fit(X_train, y_train)
                        # Forecast
                        last_values = X_train.iloc[-1].values.reshape(1, -1)
                        forecasts_list = []
                        for _ in range(periods_input):
                            y_pred = model.predict(last_values)
                            forecasts_list.append(y_pred[0])
                            # Update last_values with new prediction
                            last_values = np.roll(last_values, -1)
                            last_values[0, -1] = y_pred[0]
                    index_of_fc = pd.date_range(df_rf.index[-1], periods=periods_input+1, freq=freq_str)[1:]
                    forecasts[col] = pd.Series(forecasts_list, index=index_of_fc).to_frame(name='Forecast')
                except Exception as e:
                    st.error(f"Random Forest model failed for {col}: {e}")
                    continue
            else:
                st.error(f"Selected library {selected_library} is not supported.")
                continue

            # Calculate and display forecast generation time
            end_time = time.time()
            elapsed_time = end_time - start_time
            st.success(f"Forecast generated in {elapsed_time:.2f} seconds.")

            # Combine actuals and forecasts
            if selected_library == 'Prophet':
                forecast = forecasts[col]
                df_forecast = forecast[['yhat']]
                df_actual = data[[col]].reset_index()
                df_actual.columns = ['ds', 'y']
                df_actual = df_actual.set_index('ds')
                df_all = pd.concat([df_actual, df_forecast], axis=1)
            else:
                forecast = forecasts[col]
                df_actual = data[[col]]
                df_all = pd.concat([df_actual, forecast], axis=0)

            # Resample data if forecast frequency is different from data frequency
            df_all = df_all.asfreq(freq_str)

            # Plotting
            if selected_library == 'Prophet':
                y_col = 'y'
                yhat_col = 'yhat'
            else:
                y_col = col
                yhat_col = 'Forecast'

            fig = go.Figure()

            # Actual data
            fig.add_trace(go.Scatter(
                x=df_all.index,
                y=df_all[y_col],
                mode='lines',
                name='Actual',
                line=dict(color=line_color_actual, dash=line_style_actual)
            ))

            # Forecast data
            fig.add_trace(go.Scatter(
                x=df_all.index,
                y=df_all[yhat_col],
                mode='lines',
                name='Forecast',
                line=dict(color=line_color_forecast, dash=line_style_forecast)
            ))

            # Confidence intervals, if available and selected
            if include_confidence_interval:
                if selected_library == 'Prophet':
                    if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
                        fig.add_trace(go.Scatter(
                            x=forecast.index,
                            y=forecast['yhat_upper'],
                            mode='lines',
                            name='Upper Confidence Interval',
                            line=dict(color='rgba(255,0,0,0.2)'),
                            showlegend=False
                        ))
                        fig.add_trace(go.Scatter(
                            x=forecast.index,
                            y=forecast['yhat_lower'],
                            mode='lines',
                            name='Lower Confidence Interval',
                            fill='tonexty',
                            fillcolor='rgba(255,0,0,0.2)',
                            line=dict(color='rgba(255,0,0,0.2)'),
                            showlegend=False
                        ))
                elif selected_library == 'ARIMA/SARIMAX':
                    if 'lower' in conf_int_df.columns and 'upper' in conf_int_df.columns:
                        fig.add_trace(go.Scatter(
                            x=conf_int_df.index,
                            y=conf_int_df['upper'],
                            mode='lines',
                            name='Upper Confidence Interval',
                            line=dict(color='rgba(255,0,0,0.2)'),
                            showlegend=False
                        ))
                        fig.add_trace(go.Scatter(
                            x=conf_int_df.index,
                            y=conf_int_df['lower'],
                            mode='lines',
                            name='Lower Confidence Interval',
                            fill='tonexty',
                            fillcolor='rgba(255,0,0,0.2)',
                            line=dict(color='rgba(255,0,0,0.2)'),
                            showlegend=False
                        ))

            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Value',
                title=f"Forecast vs Actuals for {col}"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Display dataframe
            st.subheader("Forecast and Actuals")
            st.dataframe(df_all)

            # Allow user to specify time period for accuracy metrics
            st.subheader("Accuracy Metrics")
            st.write("Specify Time Period for Accuracy Metrics:")
            min_date = df_all.index.min().date()
            max_date = df_all.index.max().date()
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date, key=f"start_{col}")
            with col2:
                end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date, key=f"end_{col}")

            # Convert start_date and end_date to datetime
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)

            # Filter the data for the specified time period
            mask = (df_all.index >= start_date) & (df_all.index <= end_date)
            df_metrics = df_all.loc[mask].dropna()

            # Compute accuracy metrics
            if df_metrics.empty:
                st.warning("No data available in the selected date range.")
            else:
                if selected_library == 'Prophet':
                    y_true = df_metrics['y']
                    y_pred = df_metrics['yhat']
                else:
                    y_true = df_metrics[y_col]
                    y_pred = df_metrics[yhat_col]

                mae = mean_absolute_error(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                mape = mean_absolute_percentage_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)

                # Compute SMAPE
                def smape(A, F):
                    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

                smape_value = smape(y_true.values, y_pred.values)

                metrics = {
                    'MAE': [mae],
                    'MSE': [mse],
                    'RMSE': [rmse],
                    'MAPE': [mape],
                    'SMAPE': [smape_value],
                    'R-squared': [r2]
                }

                metrics_df = pd.DataFrame(metrics)
                st.table(metrics_df.style.format("{:.2f}"))

            # Option to download forecast data
            csv_exp = df_all.to_csv(index=True)
            st.download_button(label="Download data as CSV", data=csv_exp, file_name=f'forecast_{col}.csv', mime='text/csv')

# Footer message
st.markdown("""---""")
st.markdown(
    "<div style='text-align: center;'>Made with ❤️ by Ashwin Nair</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='text-align: center;'>If you have any issues, please contact Ashwin Nair.</div>",
    unsafe_allow_html=True
)
