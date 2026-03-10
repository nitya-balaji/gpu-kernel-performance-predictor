from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
application=Flask(__name__) #WSGI application

app=application

#route for homepage
@app.route("/") #param specifies the specific url for the web app (in string format)
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method=="GET": 
        return render_template("home.html")
    else: 
        data=CustomData(
                MWG=float(request.form.get("MWG")),
                NWG=float(request.form.get("NWG")),
                KWG=float(request.form.get("KWG")),
                MDIMC=float(request.form.get("MDIMC")),
                NDIMC=float(request.form.get("NDIMC")),
                MDIMA=float(request.form.get("MDIMA")),
                NDIMB=float(request.form.get("NDIMB")),
                KWI=float(request.form.get("KWI")),
                VWM=float(request.form.get("VWM")),
                VWN=float(request.form.get("VWN")),
                STRM=float(request.form.get("STRM")),
                STRN=float(request.form.get("STRN")),
                SA=float(request.form.get("SA")),
                SB=float(request.form.get("SB"))
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template("home.html", results=results[0])
if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True) 