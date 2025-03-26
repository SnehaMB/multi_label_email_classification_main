~#abstract class
from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    @abstractmethod
    def print_results(self, y_test, y_pred):
        pass
