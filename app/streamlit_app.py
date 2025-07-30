# app.py
import streamlit as st
import pandas as pd
import joblib
import altair as alt
from supabase import create_client, Client
import os
from dotenv import load_dotenv

load_dotenv()

# Load model and preprocessor
model = joblib.load("artifacts/model.pkl")
preprocessor = joblib.load("artifacts/preprocessor.pkl")

# Connect to Supabase
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

def load_all_data_from_supabase(supabase_url, supabase_key, table_name='transformed_customer_data'):
    supabase = create_client(supabase_url, supabase_key)

    all_rows = []
    batch_size = 1000
    offset = 0
    while True:
        response = supabase.table(table_name).select("*").range(offset, offset + batch_size - 1).execute()
        data = response.data
        if not data:
            break
        all_rows.extend(data)
        offset += batch_size
    return pd.DataFrame(all_rows)

# Streamlit page config
st.set_page_config(page_title="Customer Lifetime Value Dashboard", layout="wide")

st.markdown(
    "<h1 style='text-align: center; color: #4A90E2;'>🧠 Customer Lifetime Value Dashboard</h1>",
    unsafe_allow_html=True,
)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔎 Prediction", "📊 Analytics Dashboard", "📘 About"])

# ------------- PREDICTION TAB ----------------
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

# ------------- ANALYTICS DASHBOARD TAB ----------------
with tab2:
    st.subheader("📊 Customer Analytics")
    custom_palette = ['#87CEEB', '#1E3A8A', '#2E8B57', '#FFD700', '#FF6347']

    df = load_all_data_from_supabase(url,key)
    if not df.empty:

        # Bar Chart: Customers by Payment Method
        st.markdown("#### 🏦 Customers by Payment Method")
        method_chart = (
            alt.Chart(df)
            .mark_bar(size=40)
            .encode(
                x=alt.X("count()", title="Number of Customers"),
                y=alt.Y("payment_method:N", title="Payment Method", sort="-x"),
                color=alt.Color("payment_method:N", scale=alt.Scale(range=custom_palette)),
                tooltip=["payment_method", "count()"]
            )
            .properties(height=300)
        )
        st.altair_chart(method_chart, use_container_width=True)

        # Line Chart: Monthly Sales (assuming `purchase_date` column exists)
        if "invoice_date" in df.columns:
            df["invoice_date"] = pd.to_datetime(df["invoice_date"])
            monthly_sales = df.groupby(df["invoice_date"].dt.to_period("M"))["purchase_amount"].sum().reset_index()
            monthly_sales["invoice_date"] = monthly_sales["invoice_date"].astype(str)

            st.markdown("#### 📈 Monthly Sales Trend")
            line_chart = (
                alt.Chart(monthly_sales)
                .mark_line(color="#FF6347", point=alt.OverlayMarkDef(color="#FFD700", size=60))
                .encode(
                    x="invoice_date:T",
                    y="purchase_amount:Q",
                    tooltip=["invoice_date", "purchase_amount"]
                )
                .properties(height=300)
            )
            st.altair_chart(line_chart, use_container_width=True)

        # Pie Chart: Product Category Distribution
        st.markdown("#### 🛍️ Product Category Distribution")
        category_counts = df["product_category"].value_counts().reset_index()
        category_counts.columns = ["product_category", "count"]
        pie_chart = alt.Chart(category_counts).mark_arc(innerRadius=30).encode(
            theta="count:Q",
            color=alt.Color("product_category:N", scale=alt.Scale(range=custom_palette)),
            tooltip=["product_category", "count"]
        )
        st.altair_chart(pie_chart, use_container_width=True)

        # Top Customer Segments
        st.markdown("#### 🧑‍🤝‍🧑 Top Customer Segments")
        segment_chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("count()", title="Customers"),
            y=alt.Y("customer_segment:N", sort="-x"),
            color=alt.Color("customer_segment:N", scale=alt.Scale(range=custom_palette))
        )
        st.altair_chart(segment_chart, use_container_width=True)

    else:
        st.warning("⚠️ No data available to display analytics.")

# ------------- ABOUT TAB ----------------
with tab3:
    st.markdown("""
    ### 📘 About This App
    This app predicts **Customer Lifetime Value (CLV)** using a machine learning model and provides real-time dashboards for customer insights.

    - **Data Source**: Supabase
    - **Model**: Random Forest Regressor
    - **Interface**: FastAPI for API, Streamlit for dashboard
    - **Deployed With**: Docker, GitHub Actions, Railway

    #### Created by Sujato Dutta
    Connect on [LinkedIn](https://linkedin.com/in/sujato-dutta)
    """, unsafe_allow_html=True)
