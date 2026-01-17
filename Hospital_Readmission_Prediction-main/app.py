import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier


@st.cache_data
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    return model, feature_columns

model, feature_columns = load_model()

st.title("🏥 Hospital Readmission Prediction")

st.markdown("""
Enter patient details below to predict the likelihood of hospital readmission.
""")


input_data = {}
for feature in feature_columns:
    
    input_data[feature] = st.number_input(f"{feature}", value=0)


input_df = pd.DataFrame([input_data])


if st.button("Predict Readmission"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High risk of readmission ({prediction_proba*100:.2f}%)")
    else:
        st.success(f"✅ Low risk of readmission ({prediction_proba*100:.2f}%)")


if st.checkbox("Show input data"):
    st.write(input_df)
