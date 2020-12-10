from abc import ABC


class BaseModel(ABC):
    """
    Abstract model class
    """
    def train(self, X, y):
        pass

    def predict(self, X):
        pass

    def save(self):
        pass

    def load(self):
        pass
