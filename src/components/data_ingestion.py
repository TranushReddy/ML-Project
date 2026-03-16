import pandas as pd
import os
from sklearn.model_selection import train_test_split


class DataIngestion:

    def __init__(self):
        self.train_path = "artifacts/train.csv"
        self.test_path = "artifacts/test.csv"
        self.raw_path = "artifacts/raw.csv"

    def initiate_data_ingestion(self):

        df = pd.read_csv("data/customer_segmentation_data.csv")

        os.makedirs("artifacts", exist_ok=True)

        df.to_csv(self.raw_path, index=False)

        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

        train_set.to_csv(self.train_path, index=False)
        test_set.to_csv(self.test_path, index=False)

        return self.train_path, self.test_path
