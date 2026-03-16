from sklearn.cluster import KMeans
import pandas as pd
from src.utils import save_object


class ModelTrainer:

    def __init__(self):
        self.model_path = "artifacts/model.pkl"

    def initiate_model_trainer(self, train_array):

        model = KMeans(n_clusters=3, random_state=42)

        model.fit(train_array)

        save_object(self.model_path, model)

        return model
