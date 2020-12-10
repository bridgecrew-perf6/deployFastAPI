from typing import List

import numpy as np
from fastapi import FastAPI, Depends
from pydantic import BaseModel, ValidationError, validator
from src.model.random_forest import RandomForest

n_features = 178

class PredictRequest(BaseModel):
    data: List[List[float]]

    @validator("data")
    def check_dimensionality(cls, v):
        for point in v:
            if len(point) != n_features:
                raise ValueError(f"Must contain {n_features} features.")
        return v

class PredictResponse(BaseModel):
    data: List[float]

def get_model():
    clf = RandomForest(model_name="rf_201209")
    clf.load()
    return clf

app = FastAPI()

@app.post("/predict", response_model=PredictResponse)
async def predict(input: PredictRequest, clf: RandomForest = Depends(get_model)):
    X = np.array(input.data)
    y_pred = clf.predict(X)
    result = PredictResponse(data=y_pred.tolist())
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
    #from sklearn.datasets import load_wine
    #X, y = load_wine(return_X_y=True)
    #clf = RandomForest(model_name="rf_201209.joblib")
    #clf.load()
    #print(clf.predict(X))


