import joblib
from utils.preprocessing import load_and_preprocess_data
from utils.metrics import print_metrics
from sklearn.model_selection import train_test_split

def evaluate_model():
    # Load model
    model = joblib.load("diabetes_model.pkl")

    # Load and split data
    _, X_test, _, y_test = load_and_preprocess_data("data/diabetes.csv")
    _, X_test, _, y_test = train_test_split(X_test, y_test, test_size=0.2, random_state=42)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype("int32")  # Threshold to binary predictions

    print("Test Set Results:")
    print_metrics(y_test, y_pred)

if __name__ == "__main__":
    evaluate_model()
