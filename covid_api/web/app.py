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

scaler = pickle.load(open("data/scaler.pkl","rb"))
data_pred = pickle.load(open("data/data_pred.pkl","rb"))
dict_state_params = pickle.load(open("data/dict_state_params.pkl","rb"))
model = load_model("data/model_new.h5")

def predict(state, duration):

    df_temp = data_pred.loc[data_pred["Province_State"]==state, :].reset_index(drop=True)
    total_pop = df_temp.loc[0, "Total_pop"]
    df_future = {
                    "Date": pd.date_range(df_temp["Date"].tail(1).values[0], periods=duration+1, freq="D")[1:],
                }
    cols = df_temp.columns[4:29]
    for col in cols:
        s = df_temp.loc[:, col]
        s.index = df_temp["Date"]
        df_future[col] = STLForecast(s, AutoReg, model_kwargs={"lags": 30}).fit().forecast(duration).values

    df_future = pd.DataFrame.from_dict(df_future)
    df_future["Tested_Daily_pred"] = df_future.apply(lambda row: int(dict_state_params[state]["People_Tested_PolyCoef"]*(1+2*(row["Date"]-datetime.datetime(2020, 4, 12)).days)), axis=1)
    confirmed = df_temp["Active_pred"][-7:].values.tolist()

    for i, row in df_future.iterrows():
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
    def post(self):
        info = request.get_json()
        state = info["state"]
        duration = int(info["duration"])

        pred = predict(state, duration)
        result = {
            "result": pred,
            "Status Code": 202,
        }

        return jsonify(result)

api.add_resource(Pred, '/pred')
@app.route("/")
def hello():
    return "Hi"
if __name__ == "__main__":
    app.run(host="0.0.0.0")
