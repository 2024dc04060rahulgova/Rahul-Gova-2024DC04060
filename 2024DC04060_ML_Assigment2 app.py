import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)



# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Heart Disease ML Comparison", layout="wide")

st.title("❤️ Heart Disease Classification Model Comparison")
st.markdown("Compare multiple Machine Learning models including XGBoost.")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload Heart Dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Overview")
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

    if "target" not in df.columns:
        st.error("Dataset must contain a column named 'target'")
    else:

        # -------------------------------
        # Train Test Split
        # -------------------------------
        X = df.drop("target", axis=1)
        y = df["target"]

        test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

        # -------------------------------
        # Model Evaluation Function
        # -------------------------------
        def evaluate_model(model, X_test, y_test):
            y_pred = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.decision_function(X_test)

            return {
                "Accuracy": accuracy_score(y_test, y_pred),
                "AUC": roc_auc_score(y_test, y_prob),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred),
                "MCC": matthews_corrcoef(y_test, y_pred)
            }

        # -------------------------------
        # Train Models
        # -------------------------------
        if st.button("🚀 Train Models"):

            results = {}

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "KNN": KNeighborsClassifier(n_neighbors=5),
                "Naive Bayes": GaussianNB(),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "XGBoost": XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=4,
                    random_state=42,
                    use_label_encoder=
