import os
import sys
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i] #grab the name of current model
            model = list(models.values())[i] #grab the value for current model (e.g. LinearRegression -> LinearRegression() - grabbing this)
            para = param.get(model_name, {}) #grab hyperparameters that belong to that model
            
            #cv=3 - split training data into 3 chunks for cross-validation 
            #GridSearch will choose all provided hyperparameter combinations on these splits (.fit() - next line actually tests these combos by running them)
            gs = GridSearchCV(model, para, cv=3, n_jobs=-1, verbose=1) 
            gs.fit(X_train, y_train) #try every combo of params to find best outcome

        
            model.set_params(**gs.best_params_) #apply best set of params to the model
            model.fit(X_train, y_train) #train model on entire training dataset with the best set of params

            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred) #compare predictions with actual runtimes for test data

            report[model_name] = test_model_score #add these values to the report dictionary

        return report

    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)