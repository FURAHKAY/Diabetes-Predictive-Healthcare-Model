from utils.preprocessing import load_and_preprocess_data
from utils.metrics import print_metrics
import joblib, pandas as pd
from sklearn.model_selection import train_test_split
from utils.preprocessing import FEATURES

def evaluate_model():
    # Load model
    model = joblib.load("diabetes_model.pkl")

    # Load and split data
    df = pd.read_csv("data/diabetes.csv")
    X, y = df[FEATURES].values, df["Outcome"].values
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    print("Test Set Results:")
    print_metrics(y_test, y_pred)

if __name__ == "__main__":
    evaluate_model()
