import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

from prophet import Prophet
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from statsmodels.tsa.api import ExponentialSmoothing
from tbats import TBATS

from autots import AutoTS
from darts import TimeSeries
from darts.models import (
    ARIMA as DartsARIMA,
    ExponentialSmoothing as DartsExponentialSmoothing,
    Prophet as DartsProphet,
    Theta,
    FFT as DartsFFT,
    NaiveSeasonal,
    NaiveDrift,
    NaiveMean,
    RNNModel,
    TransformerModel,
    NBEATSModel,
    LightGBMModel,
    CatBoostModel,
    KalmanForecaster,
)
from neuralprophet import NeuralProphet

st.set_page_config(page_title="Auto Forecasting App", layout="wide")
st.title('📈 Auto Forecasting App')

st.markdown("""
This app allows you to upload time series data and perform forecasting using multiple forecasting models.
""")

# Sidebar for file upload and settings
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error("Error reading file: " + str(e))
            st.stop()

        # Select date column
        st.subheader("Select Date Column")
        date_column = st.selectbox("Date Column", data.columns)
        data[date_column] = pd.to_datetime(data[date_column], errors='coerce')

        if data[date_column].isnull().any():
            st.error("Date column contains invalid dates.")
            st.stop()

        # Select data columns
        st.subheader("Select Data Columns to Forecast")
        data_columns = st.multiselect("Data Columns", [col for col in data.columns if col != date_column])

        if not data_columns:
            st.error("Please select at least one data column to forecast.")
            st.stop()

        data.set_index(date_column, inplace=True)
        data.index.name = 'ds'  # Renaming index to 'ds' for consistency

        # Select forecasting library
        st.subheader("Select Forecasting Library")
        library_options = ['Prophet', 'ARIMA', 'Exponential Smoothing', 'TBATS', 'AutoTS', 'Darts', 'NeuralProphet']
        selected_library = st.selectbox("Forecasting Library", library_options)

        # Based on selected library, select models and parameters
        if selected_library == 'AutoTS':
            st.subheader("AutoTS Model Selection")
            autots_model_list = st.multiselect("Select Models", [
                'ARIMA', 'AverageValueNaive', 'DynamicFactor', 'ETS', 'FBProphet', 'GLM', 'GLS', 'LastValueNaive',
                'RollingRegression', 'SeasonalNaive', 'UnobservedComponents', 'VAR', 'VECM', 'WindowRegression', 'ARCH',
                'GluonTS', 'Theta', 'MotifSimulation', 'ComponentAnalysis', 'DatepartRegression', 'UnivariateMotif',
                'MultivariateMotif', 'SectionalMotif', 'MetricMotif', 'SeasonalityMotif', 'UnivariateRegression',
                'MultivariateRegression', 'VARMAX', 'DynamicFactorMQ', 'ARDL', 'MAR', 'NVAR', 'RRVAR', 'TMF', 'LATC',
                'Greykite', 'NeuralProphet', 'NeuralForecast'
            ])
            autots_params = {}
            autots_params['forecast_length'] = st.number_input("Forecast Length", min_value=1, value=14)
            autots_params['frequency'] = st.text_input("Frequency (e.g., 'D', 'W', 'M')", value='D')
            autots_params['ensemble'] = st.selectbox("Ensemble", ['simple', 'distance', 'horizontal', 'probabilistic', None])
        elif selected_library == 'Darts':
            st.subheader("Darts Model Selection")
            darts_model_options = [
                'ARIMA', 'ExponentialSmoothing', 'Prophet', 'Theta', 'FFT', 'KalmanForecaster', 'NaiveSeasonal',
                'NaiveDrift', 'NaiveMean', 'RNNModel', 'TransformerModel', 'NBEATSModel', 'LightGBMModel', 'CatBoostModel'
            ]
            darts_selected_model = st.selectbox("Select Model", darts_model_options)
            darts_params = {}
            if darts_selected_model == 'ARIMA':
                darts_params['p'] = st.number_input("ARIMA p", min_value=0, value=1)
                darts_params['d'] = st.number_input("ARIMA d", min_value=0, value=0)
                darts_params['q'] = st.number_input("ARIMA q", min_value=0, value=0)
            elif darts_selected_model == 'ExponentialSmoothing':
                darts_params['seasonal_periods'] = st.number_input("Seasonal Periods", min_value=1, value=12)
                darts_params['trend'] = st.selectbox("Trend", ['add', 'mul', None])
                darts_params['seasonal'] = st.selectbox("Seasonal", ['add', 'mul', None])
            elif darts_selected_model == 'Prophet':
                darts_params['growth'] = st.selectbox("Growth", ['linear', 'logistic'])
            elif darts_selected_model in ['RNNModel', 'TransformerModel', 'NBEATSModel']:
                darts_params['input_chunk_length'] = st.number_input("Input Chunk Length", min_value=1, value=14)
                darts_params['output_chunk_length'] = st.number_input("Output Chunk Length", min_value=1, value=7)
                darts_params['n_epochs'] = st.number_input("Number of Epochs", min_value=1, value=100)
        elif selected_library == 'NeuralProphet':
            st.subheader("NeuralProphet Model Parameters")
            np_params = {}
            np_params['seasonality_mode'] = st.selectbox("Seasonality Mode", ['additive', 'multiplicative'])
            np_params['n_forecasts'] = st.number_input("Number of Forecast Steps", min_value=1, value=1)
            np_params['n_lags'] = st.number_input("Number of Lags", min_value=0, value=0)
            # Removed num_hidden_layers and d_hidden as they are causing errors
            # np_params['num_hidden_layers'] = st.number_input("Number of Hidden Layers", min_value=0, value=0)
            # np_params['d_hidden'] = st.number_input("Hidden Layer Dimension", min_value=0, value=0)
        else:
            # Existing models from previous code
            if selected_library == 'Prophet':
                st.subheader("Prophet Model Parameters")
                prophet_params = {}
                prophet_params['growth'] = st.selectbox("Growth", ['linear', 'logistic'], index=0)
                if prophet_params['growth'] == 'logistic':
                    # Need to set 'cap' and optionally 'floor' in the data
                    st.info("Since you selected 'logistic' growth, you need to specify 'cap' and optionally 'floor' in your data.")
                    cap_value = st.number_input("Cap (capacity) value", value=1.0)
                    floor_value = st.number_input("Floor value", value=0.0)
                else:
                    cap_value = None
                    floor_value = None
                prophet_params['changepoints'] = st.text_input("Changepoints (comma-separated dates, YYYY-MM-DD)", value="")
                prophet_params['n_changepoints'] = st.number_input("Number of Changepoints", min_value=0, value=25)
                prophet_params['changepoint_range'] = st.slider("Changepoint Range", min_value=0.0, max_value=1.0, value=0.8)
                prophet_params['changepoint_prior_scale'] = st.number_input("Changepoint Prior Scale", min_value=0.001, value=0.05, format="%.3f")
                prophet_params['seasonality_prior_scale'] = st.number_input("Seasonality Prior Scale", min_value=0.01, value=10.0, format="%.2f")
                prophet_params['holidays_prior_scale'] = st.number_input("Holidays Prior Scale", min_value=0.01, value=10.0, format="%.2f")
                prophet_params['seasonality_mode'] = st.selectbox("Seasonality Mode", ['additive', 'multiplicative'], index=0)
                prophet_params['yearly_seasonality'] = st.radio("Yearly Seasonality", options=['auto', 'True', 'False', 'Custom'], index=0)
                if prophet_params['yearly_seasonality'] == 'Custom':
                    prophet_params['yearly_seasonality'] = st.number_input("Yearly Seasonality Fourier Order", min_value=1, value=10)
                elif prophet_params['yearly_seasonality'] == 'True':
                    prophet_params['yearly_seasonality'] = True
                elif prophet_params['yearly_seasonality'] == 'False':
                    prophet_params['yearly_seasonality'] = False
                else:
                    prophet_params['yearly_seasonality'] = 'auto'

                prophet_params['weekly_seasonality'] = st.radio("Weekly Seasonality", options=['auto', 'True', 'False', 'Custom'], index=0)
                if prophet_params['weekly_seasonality'] == 'Custom':
                    prophet_params['weekly_seasonality'] = st.number_input("Weekly Seasonality Fourier Order", min_value=1, value=3)
                elif prophet_params['weekly_seasonality'] == 'True':
                    prophet_params['weekly_seasonality'] = True
                elif prophet_params['weekly_seasonality'] == 'False':
                    prophet_params['weekly_seasonality'] = False
                else:
                    prophet_params['weekly_seasonality'] = 'auto'

                prophet_params['daily_seasonality'] = st.radio("Daily Seasonality", options=['auto', 'True', 'False', 'Custom'], index=0)
                if prophet_params['daily_seasonality'] == 'Custom':
                    prophet_params['daily_seasonality'] = st.number_input("Daily Seasonality Fourier Order", min_value=1, value=4)
                elif prophet_params['daily_seasonality'] == 'True':
                    prophet_params['daily_seasonality'] = True
                elif prophet_params['daily_seasonality'] == 'False':
                    prophet_params['daily_seasonality'] = False
                else:
                    prophet_params['daily_seasonality'] = 'auto'

                prophet_params['interval_width'] = st.slider("Interval Width", min_value=0.0, max_value=1.0, value=0.80)
                prophet_params['mcmc_samples'] = st.number_input("MCMC Samples", min_value=0, value=0)
                prophet_params['uncertainty_samples'] = st.number_input("Uncertainty Samples", min_value=0, value=1000)
                # Holidays
                st.subheader("Holidays")
                use_holidays = st.checkbox("Use Holidays", value=False)
                if use_holidays:
                    # The user can upload a DataFrame with holidays
                    st.info("Upload a CSV or Excel file with holidays. It should have columns 'ds' (date), 'holiday' (name), and optionally 'lower_window', 'upper_window', 'prior_scale'")
                    holidays_file = st.file_uploader("Upload Holidays File", type=["csv", "xlsx"])
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
                add_seasonality = st.checkbox("Add Custom Seasonality", value=False)
                if add_seasonality:
                    seasonality_name = st.text_input("Seasonality Name", value="monthly")
                    period = st.number_input("Period", min_value=0.0, value=30.5)
                    fourier_order = st.number_input("Fourier Order", min_value=1, value=5)
                    prior_scale = st.number_input("Prior Scale", min_value=0.01, value=10.0)
                else:
                    seasonality_name = None
            elif selected_library == 'ARIMA':
                st.subheader("ARIMA Model Parameters")
                arima_params = {}
                arima_params['start_p'] = st.number_input("Start p", min_value=0, max_value=10, value=1)
                arima_params['start_q'] = st.number_input("Start q", min_value=0, max_value=10, value=1)
                arima_params['max_p'] = st.number_input("Max p", min_value=0, max_value=10, value=3)
                arima_params['max_q'] = st.number_input("Max q", min_value=0, max_value=10, value=3)
                arima_params['m'] = st.number_input("Seasonal Period (m)", min_value=1, value=1)
                arima_params['seasonal'] = st.checkbox("Seasonal", value=False)
            elif selected_library == 'Exponential Smoothing':
                st.subheader("Exponential Smoothing Parameters")
                es_params = {}
                es_params['trend'] = st.selectbox("Trend Component", [None, 'add', 'mul'])
                es_params['seasonal'] = st.selectbox("Seasonal Component", [None, 'add', 'mul'])
                es_params['seasonal_periods'] = st.number_input("Seasonal Periods", min_value=1, value=12)
            elif selected_library == 'TBATS':
                st.subheader("TBATS Model Parameters")
                tbats_params = {}
                tbats_params['use_box_cox'] = st.checkbox("Use Box-Cox Transformation", value=False)
                tbats_params['use_trend'] = st.checkbox("Use Trend", value=True)
                tbats_params['use_damped_trend'] = st.checkbox("Use Damped Trend", value=False)
                tbats_params['sp'] = st.number_input("Seasonal Periods", min_value=1, value=12)

        # Select forecast frequency
        st.subheader("Forecast Settings")
        st.write("Select Forecast Frequency for the output:")
        forecast_frequency = st.selectbox("Forecast Frequency", ['Daily', 'Weekly', 'Monthly'])

        if forecast_frequency == 'Daily':
            freq_str = 'D'
        elif forecast_frequency == 'Weekly':
            freq_str = 'W'
        elif forecast_frequency == 'Monthly':
            freq_str = 'M'

        # Forecast horizon
        st.write("Select Forecast Horizon:")
        periods_input = st.number_input('How many periods would you like to forecast into the future?', min_value=1, value=30)

if uploaded_file is not None and data_columns:
    forecasts = {}
    tabs = st.tabs(data_columns)
    for i, col in enumerate(data_columns):
        with tabs[i]:
            st.header(f"Forecasting for **{col}**")
            df = data[[col]].copy()

            # Resample data if forecast frequency is different from data frequency
            df = df.asfreq(freq_str)
            df = df.fillna(method='ffill')  # Fill missing values

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

                if add_seasonality and seasonality_name:
                    model.add_seasonality(name=seasonality_name, period=period, fourier_order=fourier_order, prior_scale=prior_scale)

                model.fit(df_prophet)

                # For logistic growth, need to specify 'cap' and 'floor' in future dataframe
                future = model.make_future_dataframe(periods=periods_input, freq=freq_str)

                if prophet_params['growth'] == 'logistic':
                    future['cap'] = cap_value
                    future['floor'] = floor_value

                forecast = model.predict(future)
                forecasts[col] = forecast[['ds', 'yhat']].set_index('ds')
            elif selected_library == 'ARIMA':
                y = df[col]
                try:
                    model = pm.auto_arima(
                        y,
                        start_p=arima_params['start_p'],
                        start_q=arima_params['start_q'],
                        max_p=arima_params['max_p'],
                        max_q=arima_params['max_q'],
                        m=arima_params['m'],
                        seasonal=arima_params['seasonal'],
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True
                    )
                    n_periods = periods_input
                    forecast = model.predict(n_periods=n_periods)
                    index_of_fc = pd.date_range(y.index[-1], periods=n_periods+1, freq=freq_str)
                    index_of_fc = index_of_fc[1:]  # remove start date
                    fc_series = pd.Series(forecast, index=index_of_fc)
                    forecasts[col] = fc_series.to_frame(name='Forecast')
                except Exception as e:
                    st.error(f"ARIMA model failed for {col}: {e}")
                    continue
            elif selected_library == 'Exponential Smoothing':
                y = df[col]
                try:
                    model = ExponentialSmoothing(
                        y,
                        trend=es_params['trend'],
                        seasonal=es_params['seasonal'],
                        seasonal_periods=es_params['seasonal_periods']
                    )
                    model_fit = model.fit()
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
                    estimator = TBATS(
                        seasonal_periods=[tbats_params['sp']],
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
            elif selected_library == 'AutoTS':
                # Prepare data for AutoTS
                try:
                    df_autots = df.copy()
                    df_autots = df_autots.fillna(method='ffill')
                    df_autots.index.freq = freq_str  # Ensure correct frequency

                    # AutoTS requires the data to be in wide format with datetime index
                    model = AutoTS(
                        forecast_length=autots_params['forecast_length'],
                        frequency=autots_params['frequency'],
                        ensemble=autots_params['ensemble'],
                        model_list=autots_model_list if autots_model_list else 'default',
                        max_generations=5,
                        num_validations=2,
                        validation_method='backwards'
                    )
                    model.fit(df_autots)
                    prediction = model.predict()
                    forecast = prediction.forecast
                    forecasts[col] = forecast[col].to_frame(name='Forecast')
                except Exception as e:
                    st.error(f"AutoTS model failed for {col}: {e}")
                    continue
            elif selected_library == 'Darts':
                y = df[col]
                ts = TimeSeries.from_series(y)
                try:
                    if darts_selected_model == 'ARIMA':
                        model = DartsARIMA(
                            p=darts_params['p'],
                            d=darts_params['d'],
                            q=darts_params['q']
                        )
                    elif darts_selected_model == 'ExponentialSmoothing':
                        model = DartsExponentialSmoothing(
                            seasonal_periods=darts_params['seasonal_periods'],
                            trend=darts_params['trend'],
                            seasonal=darts_params['seasonal']
                        )
                    elif darts_selected_model == 'Prophet':
                        model = DartsProphet(
                            growth=darts_params['growth']
                        )
                    elif darts_selected_model == 'Theta':
                        model = Theta()
                    elif darts_selected_model == 'FFT':
                        model = DartsFFT()
                    elif darts_selected_model == 'KalmanForecaster':
                        model = KalmanForecaster()
                    elif darts_selected_model == 'NaiveSeasonal':
                        model = NaiveSeasonal()
                    elif darts_selected_model == 'NaiveDrift':
                        model = NaiveDrift()
                    elif darts_selected_model == 'NaiveMean':
                        model = NaiveMean()
                    elif darts_selected_model == 'RNNModel':
                        model = RNNModel(
                            input_chunk_length=darts_params['input_chunk_length'],
                            output_chunk_length=darts_params['output_chunk_length'],
                            n_epochs=darts_params['n_epochs']
                        )
                    elif darts_selected_model == 'TransformerModel':
                        model = TransformerModel(
                            input_chunk_length=darts_params['input_chunk_length'],
                            output_chunk_length=darts_params['output_chunk_length'],
                            n_epochs=darts_params['n_epochs']
                        )
                    elif darts_selected_model == 'NBEATSModel':
                        model = NBEATSModel(
                            input_chunk_length=darts_params['input_chunk_length'],
                            output_chunk_length=darts_params['output_chunk_length'],
                            n_epochs=darts_params['n_epochs']
                        )
                    elif darts_selected_model == 'LightGBMModel':
                        model = LightGBMModel()
                    elif darts_selected_model == 'CatBoostModel':
                        model = CatBoostModel()
                    else:
                        st.error(f"Model {darts_selected_model} not implemented.")
                        continue
                    model.fit(ts)
                    forecast = model.predict(n=periods_input)
                    forecast = forecast.pd_series()
                    forecasts[col] = forecast.to_frame(name='Forecast')
                except Exception as e:
                    st.error(f"Darts model failed for {col}: {e}")
                    continue
            elif selected_library == 'NeuralProphet':
                try:
                    df_np = df.reset_index().rename(columns={'ds': 'ds', col: 'y'})
                    model = NeuralProphet(
                        n_forecasts=np_params['n_forecasts'],
                        n_lags=np_params['n_lags'],
                        # num_hidden_layers=np_params['num_hidden_layers'],  # Removed this line
                        # d_hidden=np_params['d_hidden'],  # Removed this line
                        seasonality_mode=np_params['seasonality_mode']
                    )
                    metrics = model.fit(df_np, freq=freq_str)
                    future = model.make_future_dataframe(df_np, periods=periods_input)
                    forecast = model.predict(future)
                    forecast.set_index('ds', inplace=True)
                    forecasts[col] = forecast['yhat1'].to_frame(name='Forecast')
                except Exception as e:
                    st.error(f"NeuralProphet model failed for {col}: {e}")
                    continue
            else:
                st.error(f"Selected library {selected_library} is not supported.")
                continue

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

            fig = px.line(df_all, x=df_all.index, y=[y_col, yhat_col], labels={'value':'Value', 'index':'Date'})
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
