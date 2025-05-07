from models.neural_net import build_model
from utils.preprocessing import load_and_preprocess_data
from utils.metrics import print_metrics
import joblib

def train_model():
    X_train, X_test, y_train, y_test = load_and_preprocess_data("data/diabetes.csv")

    # Build and train model
    model = build_model(input_dim=X_train.shape[1])
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

    # Save the trained model
    model.save("diabetes_model.h5")
    joblib.dump(model, "diabetes_model.pkl")

    # Evaluate
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype("int32")  # threshold output
    print("Validation Results:")
    print_metrics(y_test, y_pred)

if __name__ == "__main__":
    train_model()
