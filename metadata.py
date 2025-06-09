MODELS_FOLDER = "models"
DATASETS_FOLDER = "datasets"
MODEL_NAME = "class_model-robert-arcaleanu"

COLUMNS_TO_DROP = ["RowNumber", "CustomerId", "Surname"]
BINARY_FEATURES = ["Gender"]
ONE_HOT_ENCODE_COLUMNS = ["Geography"]
MODEL_PARAMS = {
    "max_depth": 10,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": 42,
}
