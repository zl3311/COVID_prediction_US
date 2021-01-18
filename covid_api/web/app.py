from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import pandas as pd
import numpy as np
import pickle
import datetime
from statsmodels.tsa.api import STLForecast
from statsmodels.tsa.ar_model import AutoReg
import tensorflow.compat.v1 as tf
graph = tf.get_default_graph()
from keras.models import load_model

app = Flask(__name__)
api = Api(app)

# Load preprocessed objects
scaler = pickle.load(open("data/scaler.pkl","rb")) # the scaler to transform the original X variables before feeding to model
data_pred = pickle.load(open("data/data_pred.pkl","rb")) # the dataframe of variables of all states and time
dict_state_params = pickle.load(open("data/dict_state_params.pkl","rb")) # the dictionary of quadratic coefficients of the test case time series
model = load_model("data/model_new.h5") # neural network model of confirmed probability

def predict(state, duration):
    """Predicting future COVID confirmed numbers of a state
    Args:
        state(str): state full name to be predicted.
        duration(str): prediction duration as a string of an integer.
    Returns:
        result(List(float)): predicted future time series of the given state.
        Status Code: returned API status indicator (success or not).
    """

    df_temp = data_pred.loc[data_pred["Province_State"]==state, :].reset_index(drop=True) # Filter the data for the specific state
    total_pop = df_temp.loc[0, "Total_pop"] # Access state total population
    df_future = {
                    "Date": pd.date_range(df_temp["Date"].tail(1).values[0], periods=duration+1, freq="D")[1:],
                } # Create a dictionary of the predicted features which is further transformed into a dataframe
    cols = df_temp.columns[4:29] # Filter the feature columns
    
    for col in cols: 
        # For each column:
        # Predict the future time series of each feature independently.
        # 1. First, decompose the entire series into seasonal and trend patterns using seasonal trend decomposition
        # 2. Then, fit and forecast the overall trend time series using auto-regression model
        # 3. Lastly, add back the seasonal pattern to the forecasted trend time series
        s = df_temp.loc[:, col]
        s.index = df_temp["Date"]
        df_future[col] = STLForecast(s, AutoReg, model_kwargs={"lags": 30}).fit().forecast(duration).values 

    df_future = pd.DataFrame.from_dict(df_future) # transform dictionary into dataframe
    df_future["Tested_Daily_pred"] = df_future.apply(lambda row: int(dict_state_params[state]["People_Tested_PolyCoef"]*(1+2*(row["Date"]-datetime.datetime(2020, 4, 12)).days)), axis=1) # predict the number of tested cases using the quadratic function
    confirmed = df_temp["Active_pred"][-7:].values.tolist() # compute the 7-day rolling average of confirmed cases

    for i, row in df_future.iterrows():
        # For each future day:
        # 1. construct the average confirmed rate
        # 2. rescale the original features using the imported scalers
        # 3. predict the confirmed probability using the imported neural network model
        # 4. compute the death and recover cases of the state and minus them from the confirmed numbers
        It_r7 = np.mean(confirmed[-7:]) / total_pop
        x_ = scaler.transform(np.array([[It_r7] + row[cols].values.tolist()]))
        with graph.as_default():
            dI = int(row["Tested_Daily_pred"] * model.predict(x_)[0])
        dD = int(dI * dict_state_params[state]["death_rate"])
        dR = int(dI * dict_state_params[state]["recover_rate"])
        dI_ = dI - dD - dR
        confirmed.append(dI_ + confirmed[-1])


    return confirmed[7:]


class Pred(Resource):
    """Add prediction functionality
    """
    def post(self):
        info = request.get_json()
        state = info["state"] # extract state information from the request JSON
        duration = int(info["duration"]) # extract duration information from the request JSON

        pred = predict(state, duration) # call the prediction function and save prediction result
        result = {
            "result": pred,
            "Status Code": 202,
        } # prepare the API return result

        return jsonify(result) # jsonify the API result

api.add_resource(Pred, '/pred')
@app.route("/")
def hello():
    return "Hi"
if __name__ == "__main__":
    app.run(host="0.0.0.0")
