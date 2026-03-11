from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__) #WSGI application
app=application

VALIDATION_RULES = {
    "multiple_of_8": ["MWG", "NWG", "VWM", "VWN"]
}

#route for homepage
@app.route("/") #param specifies the specific url for the web app (in string format), which triggers the index() function automatically
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method=="GET": #user clicked a link to see the form (showing home.html - empty form)
        return render_template("home.html")
    else: #"POST" - user clicked submit button 
        try:
            input_data = request.form.to_dict()
            
            for param in VALIDATION_RULES["multiple_of_8"]:
                val = float(input_data.get(param, 0))
                if val % 8 != 0:
                    return render_template("home.html", error=f"Security Alert: {param} must be a multiple of 8.")

            #CustomData(...) takes all values inputted by user to create a CustomData object (with all inputted values as params)
            data=CustomData(
                    MWG=float(input_data["MWG"]), #grabs the value the user typed into the box labelled "MWG" in the HTML
                    NWG=float(input_data["NWG"]),
                    KWG=float(input_data["KWG"]),
                    MDIMC=float(input_data["MDIMC"]),
                    NDIMC=float(input_data["NDIMC"]),
                    MDIMA=float(input_data["MDIMA"]),
                    NDIMB=float(input_data["NDIMB"]),
                    KWI=float(input_data["KWI"]),
                    VWM=float(input_data["VWM"]),
                    VWN=float(input_data["VWN"]),
                    STRM=float(input_data["STRM"]),
                    STRN=float(input_data["STRN"]),
                    SA=float(input_data["SA"]),
                    SB=float(input_data["SB"])
            )
            pred_df=data.get_data_as_data_frame() #method under CustomData class (creates inputted values to be formatted as a 1-row dataframe)
            print(pred_df)
            predict_pipeline=PredictPipeline() #create a PredictPipeline object
            results=predict_pipeline.predict(pred_df) #find out runtime values using our predict method (input param is going to be our inputted values formatted as a 1-row dataframe)
            return render_template("home.html", results=results[0]) #page is reloaded and results (aka runtimes -> results[0] - numpy array) are now available for the user to see
        
        except Exception as e:
            return render_template("home.html", error=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)