import numpy as np
from src import ROOT_DIR
from src.model.base_model import BaseModel

class MockModel(BaseModel):
    def __init__(self, model_name: str = None):
        self._model = None
        self._model_path = ROOT_DIR / "src/model/builds"
        self._model_name = model_name
        self.trained = False

    def train(self, X: np.ndarray, y: np.ndarray):
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_instances = len(X)
        return np.random.rand(n_instances)

    def save(self):
        pass

    def load(self):
        return self