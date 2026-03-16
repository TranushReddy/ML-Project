import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils import save_object


class DataTransformation:

    def __init__(self):
        self.preprocessor_path = "artifacts/preprocessor.pkl"

    def get_preprocessor(self):

        numeric_features = [
            "income",
            "spending_score",
            "purchase_frequency",
            "last_purchase_amount",
        ]

        scaler = StandardScaler()

        preprocessor = ColumnTransformer([("num", scaler, numeric_features)])

        return preprocessor

    def initiate_data_transformation(self, train_path, test_path):

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        features = [
            "income",
            "spending_score",
            "purchase_frequency",
            "last_purchase_amount",
        ]

        X_train = train_df[features]
        X_test = test_df[features]

        preprocessor = self.get_preprocessor()

        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        save_object(self.preprocessor_path, preprocessor)

        return X_train, X_test
