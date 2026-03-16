import os
import joblib


def save_object(file_path, obj):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "wb") as file_obj:
        joblib.dump(obj, file_obj)


def load_object(file_path):
    with open(file_path, "rb") as file_obj:
        return joblib.load(file_obj)
