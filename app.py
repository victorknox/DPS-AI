# essential imports
import numpy as np
from fastapi import FastAPI, Form
import pandas as pd
from starlette.responses import HTMLResponse
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import tensorflow as tf


app = FastAPI()

@app.get('/predict', response_class=HTMLResponse)
def take_inp():
    return '''
        <h1>DPS AI Challenge</h1>
        <p> The app forecasts the number of accidents in a month based on the “Monatszahlen Verkehrsunfälle” Dataset from the München Open Data Portal. 
        </br> <a href="https://www.opengov-muenchen.de/dataset/monatszahlen-verkehrsunfaelle/resource/40094bd6-f82d-4979-949b-26c8dc00b9a7">link to the dataset </a> </p>
        <form method="post">
        <input maxlength="4" name="Year" type="text" value="2021" />
        <input maxlength="2" name="Month" type="text" value="01" />        
        <input type="submit" />
        </form>
        '''

# reading input data
file_path = "./monatszahlen2112_verkehrsunfaelle.csv"
df = pd.read_csv(file_path, sep=",", decimal=",")

X_scaler = MinMaxScaler()
Y_scaler = MinMaxScaler()


df = df.rename(columns={
    'MONATSZAHL' : 'Category',
    'AUSPRAEGUNG': 'Accident_Type',
    'JAHR'       : 'Year',
    'MONAT'      : 'Month',
    'WERT'       : 'Value',
    'VORJAHRESWERT' : 'Previous_Year_Value',
    'VERAEND_VORMONAT_PROZENT' : 'Change_Previous_Month_Percentage',
    'VERAEND_VORJAHRESMONAT_PROZENT' : 'Change_Previous_Year_Month_Percentage',
    'ZWOELF_MONATE_MITTELWERT' : 'Average_12_Months',
    })
# consider data before 2021 only
accidents = df[df["Year"] < 2021]
# consider data without summe 
accidents = accidents[accidents["Month"] != "Summe"]
le = LabelEncoder()
accidents["Category"] = le.fit_transform(accidents["Category"])
accidents["Accident_Type"] = le.fit_transform(accidents["Accident_Type"])
X_scaled = X_scaler.fit_transform(accidents[['Category', 'Accident_Type', 'Month']])
Y_scaled = Y_scaler.fit_transform(accidents[['Value']])
data_val = X_scaled[:12]
val_rescaled = data_val.reshape(1, data_val.shape[0], data_val.shape[1])

@app.get('/')
def basic_view():
    return {"WELCOME": "GO TO /docs route, or /post or send post request to /predict "}

@app.post('/predict')
def predict(Year: str, Month: str  = Form(...)):
    # load the lstm model
    lstm_model = tf.keras.models.load_model('./lstm_model')
    pred = lstm_model.predict(val_rescaled)
    pred_Inverse = Y_scaler.inverse_transform(pred)

    return {
        "prediction": int(pred_Inverse[0][0]),
    }



    