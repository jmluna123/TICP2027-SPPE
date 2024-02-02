import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import datetime
import warnings

import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import LSTM, Dense, Flatten, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")


#---------MODELS---------------

checkpoint__hour2day_path = "models/lstm_hour2day/model1.ckpt"

def create_model_lstm_hour2day():
    w_model = Sequential()
    w_model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(12, 4)))
    w_model.add(Dropout(0.5))
    w_model.add(LSTM(50, activation='relu'))
    w_model.add(Flatten())
    w_model.add(Dense(1))
    w_model.compile(optimizer='adam', loss='mse',
                    metrics=[tf.metrics.MeanAbsoluteError(),
                        tf.metrics.RootMeanSquaredError(),
                        tf.metrics.MeanAbsolutePercentageError()
                        ])
    return w_model

model = create_model_lstm_hour2day()
model.load_weights(checkpoint__hour2day_path)

def predict():
    

#------DashBoard-------------
st.set_page_config(page_title="RENO predicciones", page_icon=":chart:", layout="wide")

st.title(":bar_chart: Predicciones de producción energética")
st.markdown(
    "<style>div.block-container{padding-top:1rem;}</style>", unsafe_allow_html=True
)

fl = st.file_uploader(":file_folder: Suba los datos meteorológicos", type=['csv'])

if fl is not None:
    weather_data = pd.read_csv("weather_data_filtered.csv")
else:
    weather_data = pd.read_csv("weather_data_filtered.csv")
weather_data["date"] = pd.to_datetime(weather_data["date"])
last_date = pd.to_datetime(weather_data["date"]).max()

energy_data = pd.read_csv("reno_energiadiaria.csv")
energy_data["date"] = pd.to_datetime(energy_data["fechaDay"])

pred_data = pd.read_csv("preds.csv")
pred_data["date"] = pd.to_datetime(pred_data["fechaDay"])

col1,col2 = st.columns((2))

with col1:
    date = pd.to_datetime(st.date_input("Fecha", last_date))

weather_filter_df = weather_data[weather_data["date"] == date]

st.sidebar.header("Filtros")
measurement = st.sidebar.multiselect("Seleccione medición", ['Temperature', 'Humidity', 'WindDir', 'WindSpe'])

if not measurement:
    df2 = weather_filter_df[['time','Temperature']].melt(id_vars=['time'],var_name="Measurements", value_name="Value")
else:
    measurement.append('time')
    df2 = weather_filter_df[measurement].melt(id_vars=['time'],var_name="Measurements", value_name="Value")

pred_date = date + datetime.timedelta(days=1)

pred_val = pred_data[pred_data['date'] == pred_date]['generacionDiaria'].values[0]
org_val = energy_data[energy_data['date'] == pred_date]['generacionDiaria'].values[0]

col1,col2 = st.columns((2))
with col1:
    st.header("Registro siguiente día")
    st.subheader("{:.1f} (kWh)".format(org_val))
with col2:
    st.header("Predicción siguiente día")
    st.subheader("{:.1f} (kWh)".format(pred_val))


st.subheader("Mediciones")
fig = px.line(df2, x="time", y="Value", color='Measurements')
st.plotly_chart(fig, use_container_width=True)



