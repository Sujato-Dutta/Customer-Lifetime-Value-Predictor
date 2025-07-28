# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessor
model = joblib.load("artifacts/model.pkl")
preprocessor = joblib.load("artifacts/preprocessor.pkl")

# Page configuration
st.set_page_config(page_title="Customer Lifetime Value Predictor", layout="wide")
st.markdown(
    "<h1 style='text-align: center; color: #4A90E2;'>🧠 Customer Lifetime Value Prediction</h1>",
    unsafe_allow_html=True,
)

# --- Tabs Layout ---
tab1, tab2 = st.tabs(["🔎 Prediction", "📘 About This App"])

with tab1:
    st.subheader("🔍 Enter Customer Details")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        purchase_amount = st.number_input("💰 Purchase Amount", value=100.0)
        product_category = st.selectbox("📦 Product Category", ['electronics', 'books', 'home', 'clothing', 'beauty'])
        payment_method = st.selectbox("💳 Payment Method", ['credit card', 'debit card', 'upi', 'cash on delivery'])
        customer_segment = st.selectbox("👥 Customer Segment", ['new', 'returning', 'loyal'])
        region = st.selectbox("🌍 Region", ['north', 'south', 'east', 'west'])
        predict_button = st.button("🚀 Predict CLV")

    if predict_button:
        input_data = {
            "purchase_amount": purchase_amount,
            "product_category": product_category,
            "payment_method": payment_method,
            "customer_segment": customer_segment,
            "region": region
        }

        input_df = pd.DataFrame([input_data])
        transformed = preprocessor.transform(input_df)
        prediction = model.predict(transformed)[0]

        st.markdown("---")
        st.success("Prediction completed successfully!")

        st.markdown(
            f"<h2 style='text-align: center; color: #4CAF50;'>💡 Predicted CLV: ₹ {prediction:.2f}</h2>",
            unsafe_allow_html=True
        )

with tab2:
    st.markdown("""
    ### 📘 About
    This app uses a trained machine learning model to estimate **Customer Lifetime Value (CLV)** based on user inputs such as:

    - Purchase behavior
    - Region and payment preferences
    - Customer segment

    **Model**: Random Forest Regressor  
    **Data Source**: Stored and managed in Supabase  
    **Prediction Interface**: FastAPI + Streamlit  

    Created by Sujato Dutta  
    """)

# Footer
st.markdown(
    "<hr><p style='text-align: center; color: gray;'>© 2025 | CLV Predictor Dashboard</p>",
    unsafe_allow_html=True,
)
