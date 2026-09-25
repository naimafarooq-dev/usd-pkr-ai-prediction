import joblib

from river import preprocessing
from river import forest

from config import MODEL_FILE
from features import create_features, FEATURE_COLUMNS


def create_model():
    """
    Create a River online Random Forest model.
    """

    return (
        preprocessing.StandardScaler()
        | forest.ARFRegressor(
            n_models=10,
            seed=42
        )
    )


def row_to_features(row):
    """
    Convert one pandas row into a dictionary
    that River can use.
    """

    return {
        feature: float(row[feature])
        for feature in FEATURE_COLUMNS
    }


def train_initial_model(df):
    """
    Train the model using historical data.
    """

    model = create_model()

    feature_data = create_features(df)

    feature_data = feature_data.dropna(
        subset=FEATURE_COLUMNS
    )

    for _, row in feature_data.iterrows():

        x = row_to_features(row)
        y = float(row["exchange_rate"])

        model.learn_one(x, y)

    return model


def save_model(model):
    """
    Save trained model to disk.
    """

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_FILE
    )


def load_model():
    """
    Load existing trained model.
    """

    if MODEL_FILE.exists():
        return joblib.load(MODEL_FILE)

    return None


def predict(model, feature_row):
    """
    Predict USD/PKR rate.
    """

    x = row_to_features(feature_row)

    return model.predict_one(x)


def learn(model, feature_row, actual):
    """
    Update model after actual rate becomes available.
    """

    x = row_to_features(feature_row)

    model.learn_one(
        x,
        float(actual)
    )