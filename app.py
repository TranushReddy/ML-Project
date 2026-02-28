import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ======================================================
# Page Configuration
# ======================================================
st.set_page_config(
    page_title="Customer Segmentation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================
# Title
# ======================================================
st.title("👥 Customer Segmentation using K-Means")
st.markdown(
    "Segment customers based on **credit card usage behavior** using **K-Means clustering**."
)

# ======================================================
# Sidebar Configuration
# ======================================================
st.sidebar.header("⚙️ Model Configuration")

n_clusters = st.sidebar.slider(
    "Number of Clusters (K)", min_value=2, max_value=10, value=4
)
max_iter = st.sidebar.slider("Max Iterations", 100, 500, 300, 50)

# ======================================================
# Data Loading
# ======================================================
st.sidebar.header("📁 Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
use_sample = st.sidebar.checkbox("Use sample dataset", value=True)

if use_sample and uploaded_file is None:
    try:
        df = pd.read_csv("Customer_Data (1).csv")
        st.sidebar.success("✓ Sample dataset loaded")
    except FileNotFoundError:
        st.error("Dataset file not found. Please upload a CSV file.")
        st.stop()
elif uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✓ File uploaded successfully")
else:
    st.error("Please upload a dataset or enable sample data")
    st.stop()

# ======================================================
# Data Preprocessing
# ======================================================
st.sidebar.header("⚙️ Data Processing")
with st.sidebar.expander("Processing Steps"):
    st.write("1. Drop customer ID")
    st.write("2. Handle missing values")
    st.write("3. Scale numerical features")
    st.write("4. Train K-Means model")
    st.write("5. Assign cluster labels")

# Drop ID column
if "CUST_ID" in df.columns:
    df_model = df.drop("CUST_ID", axis=1)
else:
    df_model = df.copy()

# Handle missing values
df_model["MINIMUM_PAYMENTS"].fillna(df_model["MINIMUM_PAYMENTS"].mean(), inplace=True)
df_model["CREDIT_LIMIT"].fillna(df_model["CREDIT_LIMIT"].mean(), inplace=True)

# Scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_model)
scaled_df = pd.DataFrame(scaled_data, columns=df_model.columns)

# ======================================================
# Train K-Means Model
# ======================================================
kmeans = KMeans(
    n_clusters=n_clusters,
    max_iter=max_iter,
    random_state=42,
)
kmeans.fit(scaled_df)

clusters = kmeans.predict(scaled_df)
scaled_df["Cluster"] = clusters

# Evaluation
sil_score = silhouette_score(scaled_df.drop("Cluster", axis=1), clusters)

# ======================================================
# Tabs
# ======================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "📈 Visualizations", "📋 Cluster Details", "📥 Export"]
)

# ======================================================
# Tab 1: Overview
# ======================================================
with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Customers", len(df))
    with col2:
        st.metric("Clusters (K)", n_clusters)
    with col3:
        st.metric("Silhouette Score", f"{sil_score:.3f}")

    st.divider()

    st.subheader("Dataset Preview")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Original Data**")
        st.dataframe(df.head(), use_container_width=True)

    with col2:
        st.write("**With Cluster Labels**")
        preview_df = df.copy()
        preview_df["Cluster"] = clusters
        st.dataframe(preview_df.head(), use_container_width=True)

# ======================================================
# Tab 2: Visualizations
# ======================================================
with tab2:
    st.subheader("Cluster Visualizations")

    numeric_cols = scaled_df.columns.drop("Cluster").tolist()

    col1, col2 = st.columns(2)
    with col1:
        x_feat = st.selectbox("X-axis Feature", numeric_cols)
    with col2:
        y_feat = st.selectbox(
            "Y-axis Feature", numeric_cols, index=1 if len(numeric_cols) > 1 else 0
        )

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        scaled_df[x_feat],
        scaled_df[y_feat],
        c=clusters,
        cmap="viridis",
        alpha=0.7,
    )
    ax.set_xlabel(x_feat, fontweight="bold")
    ax.set_ylabel(y_feat, fontweight="bold")
    ax.set_title("Customer Clusters (2D View)", fontweight="bold")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, use_container_width=True)

    # 3D Interactive Plot
    st.subheader("3D Interactive Cluster View")

    if len(numeric_cols) >= 3:
        col1, col2, col3 = st.columns(3)
        with col1:
            x3 = st.selectbox("X-axis", numeric_cols, key="x3")
        with col2:
            y3 = st.selectbox("Y-axis", numeric_cols, key="y3", index=1)
        with col3:
            z3 = st.selectbox("Z-axis", numeric_cols, key="z3", index=2)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=scaled_df[x3],
                y=scaled_df[y3],
                z=scaled_df[z3],
                mode="markers",
                marker=dict(
                    size=4,
                    color=clusters,
                    colorscale="Viridis",
                    opacity=0.8,
                ),
            )
        )

        fig.update_layout(
            title="3D Customer Segmentation",
            scene=dict(
                xaxis_title=x3,
                yaxis_title=y3,
                zaxis_title=z3,
            ),
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

# ======================================================
# Tab 3: Cluster Details
# ======================================================
with tab3:
    st.subheader("Cluster Statistics")

    cluster_summary = (
        pd.concat([df_model, pd.Series(clusters, name="Cluster")], axis=1)
        .groupby("Cluster")
        .mean()
        .round(2)
    )

    st.dataframe(cluster_summary, use_container_width=True)

    st.divider()
    st.subheader("Cluster Sizes")

    cluster_counts = pd.Series(clusters).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(cluster_counts.index, cluster_counts.values, color="steelblue")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customers per Cluster")
    ax.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig, use_container_width=True)

# ======================================================
# Tab 4: Export
# ======================================================
with tab4:
    st.subheader("Export Results")

    results_df = df.copy()
    results_df["Cluster"] = clusters

    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Clustered Data (CSV)",
        data=csv,
        file_name="customer_segmentation_results.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Summary Report")

    summary = f"""
    ## Customer Segmentation Report

    ### Dataset Overview
    - Total Customers: {len(df)}
    - Number of Clusters (K): {n_clusters}

    ### Model Details
    - Algorithm: K-Means Clustering
    - Max Iterations: {max_iter}
    - Silhouette Score: {sil_score:.3f}

    ### Purpose
    This model segments customers based on spending behavior,
    credit usage, and payment patterns to support business decisions.
    """

    st.markdown(summary)

    st.download_button(
        label="📥 Download Report (TXT)",
        data=summary,
        file_name="customer_segmentation_report.txt",
        mime="text/plain",
    )

# ======================================================
# Footer
# ======================================================
st.divider()
st.caption("📊 Customer Segmentation System | Powered by Streamlit & Machine Learning")
