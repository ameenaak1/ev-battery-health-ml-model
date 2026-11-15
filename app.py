# EV Battery Health Prediction - Streamlit app
# Run with:  streamlit run app.py

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="EV Battery Health", layout="wide")
st.title("EV Battery Health Prediction")
st.write("Choose a value for each input (or pick **Enter manually** to type your own), then click **Predict**.")


# ---------------- load the 3 saved models ----------------
@st.cache_resource
def load_models():
    soh = joblib.load("models/soh_model.pkl")
    rul = joblib.load("models/rul_model.pkl")
    risk = joblib.load("models/failure_risk_model.pkl")
    return soh, rul, risk


# ---------------- load the dataset (only to make the 3 default values) ----------------
@st.cache_data
def load_data():
    return pd.read_csv("dataset/ev_battery_health.csv")


soh_saved, rul_saved, risk_saved = load_models()
df = load_data()

columns = soh_saved["columns"]               # input columns, same for all 3 models
text_columns = list(soh_saved["encoders"])   # columns that contain text (chemistry, manufacturer, ...)


# ---------------- input boxes ----------------
user_input = {}
box = st.columns(3)   # show the inputs in 3 columns

for i, col in enumerate(columns):
    with box[i % 3]:

        if col in text_columns:
            # text column -> choose one of the known values
            options = list(soh_saved["encoders"][col].classes_)
            user_input[col] = st.selectbox(col, options)

        elif col == "recall_flag":
            # yes / no column
            user_input[col] = st.selectbox(col, [0, 1], format_func=lambda v: "Yes (1)" if v == 1 else "No (0)")

        else:
            # number column -> 3 easy values (low / normal / high) + manual entry
            low = round(float(df[col].quantile(0.25)), 2)
            normal = round(float(df[col].quantile(0.50)), 2)
            high = round(float(df[col].quantile(0.75)), 2)

            choice = st.selectbox(col, [low, normal, high, "Enter manually"], index=1)

            if choice == "Enter manually":
                user_input[col] = st.number_input(f"Type {col}", value=float(normal), key=col + "_manual")
            else:
                user_input[col] = choice


# ---------------- prepare the input like in training ----------------
def prepare(saved, user_input):
    row = pd.DataFrame([user_input])

    # text -> numbers with the same LabelEncoder used in training
    for col, le in saved["encoders"].items():
        row[col] = le.transform(row[col])

    row = row[saved["columns"]]            # same column order as training
    return saved["scaler"].transform(row)  # same scaling as training


# ---------------- predict ----------------
st.divider()

if st.button("Predict", type="primary"):
    soh = soh_saved["model"].predict(prepare(soh_saved, user_input))[0]
    rul = rul_saved["model"].predict(prepare(rul_saved, user_input))[0]
    risk = risk_saved["model"].predict(prepare(risk_saved, user_input))[0]

    result = st.columns(3)
    result[0].metric("State of Health", f"{soh:.1f} %")
    result[1].metric("Remaining Useful Life", f"{max(rul, 0):.0f} cycles")
    result[2].metric("Failure Risk", risk)

    if risk == "High":
        st.error("High failure risk: this battery needs a check soon.")
    elif risk == "Medium":
        st.warning("Medium failure risk: keep an eye on this battery.")
    else:
        st.success("Low failure risk: the battery is in good condition.")
