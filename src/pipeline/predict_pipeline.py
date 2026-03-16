import pandas as pd
from src.utils import load_object


class PredictPipeline:

    def __init__(self):
        self.model = load_object("artifacts/model.pkl")
        self.preprocessor = load_object("artifacts/preprocessor.pkl")

    def predict(self, data):

        df = pd.DataFrame([data])

        features = [
            "income",
            "spending_score",
            "purchase_frequency",
            "last_purchase_amount",
        ]

        df = df[features]

        scaled = self.preprocessor.transform(df)

        prediction = self.model.predict(scaled)

        return prediction
