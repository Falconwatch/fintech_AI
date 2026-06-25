import streamlit as st


st.set_page_config(page_title="Fraud Detection Demo", layout="centered")

st.title("Fraud Detection Demo")
st.write("Starter UI for the IEEE-CIS antifraud project.")

amount = st.number_input("Transaction amount", min_value=0.0, value=100.0)
product = st.selectbox("Product code", ["W", "C", "R", "H", "S"])
device_type = st.selectbox("Device type", ["desktop", "mobile", "missing"])
email_match = st.selectbox("Email domains match", ["yes", "no", "unknown"])

if st.button("Predict"):
    heuristic_score = min(
        0.95,
        0.15 + amount / 5000 + (0.1 if device_type == "missing" else 0.0),
    )
    st.metric("Fraud probability", f"{heuristic_score:.2%}")
    if heuristic_score >= 0.75:
        st.write("Decision: `block`")
    elif heuristic_score >= 0.40:
        st.write("Decision: `review`")
    else:
        st.write("Decision: `approve`")

    st.write("Prototype note: this screen is a placeholder until the trained model is connected.")
