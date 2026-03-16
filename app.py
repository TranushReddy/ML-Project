import streamlit as st
import pandas as pd
import plotly.express as px
from src.pipeline.predict_pipeline import PredictPipeline

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("Customer Segmentation Dashboard")

# Load dataset
df = pd.read_csv("data/customer_segmentation_data.csv")

# -------------------------
# DATA OVERVIEW
# -------------------------

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(df))
col2.metric("Average Income", round(df["income"].mean(), 2))
col3.metric("Average Spending Score", round(df["spending_score"].mean(), 2))

st.dataframe(df.head())

# -------------------------
# GRAPH 1 : Income Distribution
# -------------------------

st.subheader("Income Distribution")

fig = px.histogram(df, x="income", nbins=30, title="Customer Income Distribution")

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# GRAPH 2 : Spending Score Distribution
# -------------------------

st.subheader("Spending Score Distribution")

fig2 = px.histogram(
    df, x="spending_score", nbins=30, title="Spending Score Distribution"
)

st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# GRAPH 3 : Income vs Spending
# -------------------------

st.subheader("Income vs Spending Score")

fig3 = px.scatter(
    df, x="income", y="spending_score", color="gender", title="Income vs Spending Score"
)

st.plotly_chart(fig3, use_container_width=True)

# -------------------------
# CUSTOMER SEGMENT PREDICTION
# -------------------------

st.subheader("Predict Customer Segment")

income = st.number_input("Income")
spending_score = st.number_input("Spending Score")
purchase_frequency = st.number_input("Purchase Frequency")
last_purchase_amount = st.number_input("Last Purchase Amount")

if st.button("Predict Segment"):

    data = {
        "income": income,
        "spending_score": spending_score,
        "purchase_frequency": purchase_frequency,
        "last_purchase_amount": last_purchase_amount,
    }

    pipeline = PredictPipeline()

    prediction = pipeline.predict(data)

    segment_map = {
        0: "Low Spending Customer",
        1: "Medium Spending Customer",
        2: "High Spending Customer",
    }

    st.success(f"Predicted Segment: {segment_map[int(prediction[0])]}")
from src.utils import load_object

model = load_object("artifacts/model.pkl")
preprocessor = load_object("artifacts/preprocessor.pkl")

features = df[
    ["income", "spending_score", "purchase_frequency", "last_purchase_amount"]
]

scaled = preprocessor.transform(features)

df["cluster"] = model.predict(scaled)

st.subheader("Customer Segments")

fig4 = px.scatter(
    df,
    x="income",
    y="spending_score",
    color=df["cluster"].astype(str),
    title="Customer Segmentation Clusters",
)

st.plotly_chart(fig4, use_container_width=True)
