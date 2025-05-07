import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pandas as pd
import seaborn as sns
from utils.preprocessing import load_and_preprocess_data
from utils.metrics import print_metrics
from models.neural_net import build_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

def main():
    # Load and preprocess data
    df = pd.read_csv("data/diabetes.csv")
    X_train, X_test, y_train, y_test = load_and_preprocess_data("data/diabetes.csv")
    # Load full DataFrame for plotting

    # Feature relationship plots
    features_to_plot = [
        ("Glucose", "glucose_vs_outcome.png"),
        ("BloodPressure", "bp_vs_outcome.png"),
        ("BMI", "bmi_vs_outcome.png"),
        ("Age", "age_vs_outcome.png"),
        ("Insulin", "insulin_vs_outcome.png"),
        ("DiabetesPedigreeFunction", "pedigree_vs_outcome.png"),
    ]

    for feature, filename in features_to_plot:
        plt.figure(figsize=(8,6))
        sns.boxplot(x="Outcome", y=feature, data=df)
        plt.title(f"{feature} by Diabetes Outcome")
        plt.xlabel("Diabetes (0 = No, 1 = Yes)")
        plt.ylabel(feature)
        plt.tight_layout()
        plt.savefig(f"plots/{filename}")
        plt.close()

    # Build and train model
    model = build_model(input_dim=X_train.shape[1])
    history = model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype("int32")
    print("\nTest Set Results:")
    print_metrics(y_test, y_pred)

    # Plot training history
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['loss'], label='loss')
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Accuracy and Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/training_metrics.png")  # Save the plot
    plt.close()  # Close the plot to prevent overlap if rerun

    # Confusion Matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Diabetic", "Diabetic"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("plots/confusion_matrix.png")
    plt.close()

    # Predict probabilities for ROC
    y_prob = model.predict(X_test).ravel()
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("plots/roc_curve.png")
    plt.close()


    # Predict on a new sample
    new_patient = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])  # Example
    prediction = model.predict(new_patient)
    print("\nNew Patient Prediction:", "Diabetic" if prediction > 0.5 else "Not Diabetic")

if __name__ == "__main__":
    main()
