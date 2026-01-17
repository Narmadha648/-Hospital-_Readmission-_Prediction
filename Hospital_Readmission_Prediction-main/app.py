import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hospital Readmission Prediction", layout="centered")

st.title("🏥 Hospital Readmission Prediction")
st.markdown("Enter patient details below to predict the likelihood of hospital readmission.")


@st.cache_data
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    return model, feature_columns, label_encoders

model, feature_columns, label_encoders = load_model()


input_data = {}

for feature in feature_columns:
    if feature in label_encoders:  
        options = list(label_encoders[feature].classes_)
        input_data[feature] = st.selectbox(f"{feature}", options)
    else:  
        input_data[feature] = st.number_input(f"{feature}", value=0)


input_df = pd.DataFrame([input_data])


for col, le in label_encoders.items():
    input_df[col] = le.transform(input_df[col].astype(str))

if st.button("Predict Readmission"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

 
    threshold = 0.5  
    if prediction_proba > threshold:
        st.error(f"⚠️ High risk of readmission ({prediction_proba*100:.2f}%)")
    else:
        st.success(f"✅ Low risk of readmission ({prediction_proba*100:.2f}%)")

    st.subheader("Readmission Probability Gauge")
    st.progress(prediction_proba)


    st.subheader("Feature Importance")
    importance = model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False).head(15) 

    
    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(fi_df["Feature"], fi_df["Importance"], color='skyblue')
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Top 15 Important Features")
    st.pyplot(fig)


if st.checkbox("Show input data"):
    st.write(input_df)
