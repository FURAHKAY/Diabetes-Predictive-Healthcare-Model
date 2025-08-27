# from models.neural_net import build_model
# from utils.preprocessing import load_and_preprocess_data
# from utils.metrics import print_metrics
# import joblib
#
# def train_model():
#     X_train, X_test, y_train, y_test = load_and_preprocess_data("data/diabetes.csv")
#
#     # Build and train model
#     model = build_model(input_dim=X_train.shape[1])
#     model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)
#
#     # Save the trained model
#     model.save("diabetes_model.h5")
#     joblib.dump(model, "diabetes_model.pkl")
#
#     #for inference
#     scaler = fit_scaler_on_training("data/diabetes.csv")
#     joblib.dump(scaler, "utils/scaler.pkl")
#     # Evaluate
#     y_pred = model.predict(X_test)
#     y_pred = (y_pred > 0.5).astype("int32")  # threshold output
#     print("Validation Results:")
#     print_metrics(y_test, y_pred)


# train_sklearn.py
import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from utils.preprocessing import FEATURES

def train_model():
    df = pd.read_csv("data/diabetes.csv")
    X, y = df[FEATURES].values, df["Outcome"].values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500))
    ])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(Xtr, ytr)

    yprob = pipe.predict_proba(Xte)[:, 1]
    ypred = (yprob >= 0.5).astype(int)
    print("AUC:", roc_auc_score(yte, yprob))
    print(classification_report(yte, ypred))

    joblib.dump(pipe, "diabetes_model.pkl")
    print("Saved -> diabetes_model.pkl")


if __name__ == "__main__":
    train_model()
