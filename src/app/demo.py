from pathlib import Path

import streamlit as st

from src.models.predict import load_bundle, predict_payload


st.set_page_config(page_title="Антифрод demo", layout="centered")

st.title("Антифрод demo по IEEE-CIS")
st.write("Интерфейс для показа предсказания обученной модели на одной транзакции.")

model_path = Path("models/fraud_model.joblib")
if not model_path.exists():
    st.warning(
        "Обученная модель пока не найдена. Сначала запустите `python -m src.models.train --data-dir data/raw`."
    )
    st.stop()

bundle = load_bundle(model_path)

with st.form("prediction_form"):
    amount = st.number_input("Сумма транзакции", min_value=0.0, value=120.0)
    product = st.selectbox("ProductCD", ["W", "C", "R", "H", "S"])
    card1 = st.number_input("card1", min_value=0.0, value=1500.0)
    addr1 = st.number_input("addr1", min_value=0.0, value=315.0)
    p_email = st.text_input("P_emaildomain", value="gmail.com")
    r_email = st.text_input("R_emaildomain", value="gmail.com")
    device_type = st.selectbox("DeviceType", ["desktop", "mobile", "missing"])
    device_info = st.text_input("DeviceInfo", value="Windows")
    submitted = st.form_submit_button("Предсказать")

if submitted:
    payload = {
        "TransactionAmt": amount,
        "ProductCD": product,
        "card1": card1,
        "addr1": addr1,
        "P_emaildomain": p_email,
        "R_emaildomain": r_email,
        "DeviceType": device_type,
        "DeviceInfo": device_info,
    }
    result = predict_payload(payload, bundle)
    st.metric("Вероятность мошенничества", f"{result['fraud_probability']:.2%}")
    st.write(f"Решение системы: `{result['decision']}`")
    st.caption(
        "Для финального видео можно дополнить экран текстом про порог, ручную проверку и топ-факторы риска."
    )
