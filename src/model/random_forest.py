import joblib
import numpy as np
from pathlib import Path
from src import ROOT_DIR

from src.model.base_model import BaseModel
from sklearn.ensemble import RandomForestClassifier

class RandomForest(BaseModel):
    def __init__(self, model_name: str = None):
        self._model = None
        self._model_path = ROOT_DIR / "src/model/builds"
        self._model_name = model_name
        self.trained = False

    def train(self, X: np.ndarray, y: np.ndarray):
        self._model = RandomForestClassifier()
        self._model.fit(X, y)
        self.trained = True

    def predict(self, X:np.ndarray) -> np.ndarray:
        assert self.trained, "Model not trained. Please train model first."
        return self._model.predict(X)

    def save(self):
        assert self.trained, "Model not trained. Please train model first."
        joblib.dump(self._model, self._model_path / self._model_name)

    def load(self):
        model = self._model_path / self._model_name
        assert model.exists(), "Trained model does not exist."
        self._model = joblib.load(model)
        self.trained = True

if __name__ == "__main__":
    from sklearn.datasets import load_wine
    X, y = load_wine(return_X_y=True)
    clf = RandomForest(model_name="rf_201209.joblib")
    clf.train(X, y)
    clf.save()

