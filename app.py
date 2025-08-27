import streamlit as st, numpy as np, joblib, os
from utils.preprocessing import get_feature_order

FEATURES = get_feature_order()
MODEL_PKL = "diabetes_model.pkl"

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺")
st.title("🩺 Diabetes Risk Predictor")
st.write("Scikit-learn classifier estimating diabetes probability from clinical features (Pima dataset).")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PKL):
        st.error("Model file not found. Train it first with train_sklearn.py")
        st.stop()
    return joblib.load(MODEL_PKL)

model = load_model()

defaults = {"Pregnancies":1,"Glucose":120,"BloodPressure":70,"SkinThickness":20,
            "Insulin":85,"BMI":28.5,"DiabetesPedigreeFunction":0.45,"Age":33}

cols = st.columns(2)
vals = []
for i, f in enumerate(FEATURES):
    with cols[i % 2]:
        if f in ["Pregnancies","Age"]:
            v = st.number_input(f, value=int(defaults[f]), step=1)
        else:
            v = st.number_input(f, value=float(defaults[f]))
        vals.append(v)

X = np.array(vals, dtype=float).reshape(1, -1)

if st.button("Predict"):
    prob = float(model.predict_proba(X)[0, 1])
    st.success(f"Estimated probability of diabetes: **{prob:.2%}**")
    st.progress(min(max(prob, 0.0), 1.0))

st.caption("Model & demo: Furaha Kabeya • DiabetesPredictiveHealthcare")
