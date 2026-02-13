#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1

import pandas as pd
import numpy as np


# Load dataset
df = pd.read_csv("/Users/rahulgova/Desktop/heart dataset.csv")  


print("Dataset Shape:", df.shape)
print("\nColumn Names:\n", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nSample Data:\n", df.head())




# In[2]:


#2 
X = df.drop("target", axis=1)
y = df["target"]


# In[3]:


#3 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[4]:


#4

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef


# In[5]:


#5
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    # Probability for AUC
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
    


# In[6]:


#6 LR
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_results = evaluate_model(lr, X_test, y_test)


# In[7]:


# SVD
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_results = evaluate_model(dt, X_test, y_test)


# In[8]:


#KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_results = evaluate_model(knn, X_test, y_test)


# In[9]:


#NBG
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_results = evaluate_model(nb, X_test, y_test)


# In[10]:


#RF
rf = RandomForestClassifier(
    n_estimators=100, random_state=42
)
rf.fit(X_train, y_train)
rf_results = evaluate_model(rf, X_test, y_test)


# In[11]:


# Cmparision of Results 


results_df = pd.DataFrame({
    "Logistic Regression": lr_results,
    "Decision Tree": dt_results,
    "KNN": knn_results,
    "Naive Bayes": nb_results,
    "Random Forest": rf_results,

    
}).T

print(results_df)


# In[12]:


import pandas as pd
import matplotlib.pyplot as plt

# Create DataFrame from results
results_df = pd.DataFrame({
    "Accuracy": [0.727513, 0.976190, 0.801587, 0.708995, 0.978836],
    "AUC": [0.831773, 0.976256, 0.788055, 0.789835, 0.998542],
    "Precision": [0.706667, 0.979487, 0.795122, 0.682203, 0.974747],
    "Recall": [0.811224, 0.974490, 0.831633, 0.821429, 0.984694],
    "F1 Score": [0.755344, 0.976982, 0.812968, 0.745370, 0.979695],
    "MCC": [0.456640, 0.952338, 0.602614, 0.422327, 0.957651]
}, index=[
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest"
])

# Plot comparison graph
plt.figure(figsize=(12,6))
results_df.plot(kind="bar")
plt.title("Comparison of Classification Models Across Evaluation Metrics")
plt.ylabel("Score")
plt.xlabel("Models")
plt.xticks(rotation=45, ha="right")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()


# In[13]:


"""1.1 Logistic Regression
Shows moderate accuracy and AUC
Performs well in recall but lower MCC indicates weaker overall correlation
Suitable as a baseline linear model
1.2 Decision Tree
Very high accuracy, F1-score, and MCC
Captures complex patterns effectively
Risk of overfitting, but performance is strong on this dataset
1.3 K-Nearest Neighbors (KNN)
Balanced performance across all metrics
Works well for local patterns
Computationally expensive for large datasets
1.4  Naive Bayes
Fast and simple model
Good recall but lower precision and MCC
Assumption of feature independence limits performance
1.5  Random Forest
Best overall performer
Highest AUC and MCC, indicating strong and stable classification
Ensemble approach reduces overfitting and improves generalization

 2. Conclusion: Random Forest outperforms all other models,
 followed closely by Decision Tree, 
 while Logistic Regression and Naive Bayes serve as baseline classifiers."""


# In[14]:


import joblib

# RF Model
joblib.dump(rf, "heart_model.pkl")


# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
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
st.markdown("Compare multiple Machine Learning models for heart disease prediction.")

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
        # Data Preprocessing Options
        # -------------------------------
        st.sidebar.header("Preprocessing Options")
        normalize = st.sidebar.checkbox("Normalize Features", value=False)
        
        # -------------------------------
        # Train Test Split
        # -------------------------------
        X = df.drop("target", axis=1)
        y = df["target"]
        
        if normalize:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            st.info("Features have been normalized")

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
                try:
                    y_prob = model.decision_function(X_test)
                except:
                    y_prob = y_pred  # Fallback if no probability method available

            return {
                "Accuracy": accuracy_score(y_test, y_pred),
                "AUC": roc_auc_score(y_test, y_prob),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred),
                "MCC": matthews_corrcoef(y_test, y_pred)
            }

        # -------------------------------
        # Model Selection
        # -------------------------------
        st.sidebar.header("Model Selection")
        use_logreg = st.sidebar.checkbox("Logistic Regression", value=True)
        use_dt = st.sidebar.checkbox("Decision Tree", value=True)
        use_knn = st.sidebar.checkbox("KNN", value=True)
        use_nb = st.sidebar.checkbox("Naive Bayes", value=True)
        use_rf = st.sidebar.checkbox("Random Forest", value=True)

        # -------------------------------
        # Train Models
        # -------------------------------
        if st.button("🚀 Train Models"):
            results = {}
            models = {}
            
            if use_logreg:
                models["Logistic Regression"] = LogisticRegression(max_iter=1000, random_state=42)
            if use_dt:
                models["Decision Tree"] = DecisionTreeClassifier(random_state=42)
            if use_knn:
                models["KNN"] = KNeighborsClassifier(n_neighbors=5)
            if use_nb:
                models["Naive Bayes"] = GaussianNB()
            if use_rf:
                models["Random Forest"] = RandomForestClassifier(n_estimators=100, random_state=42)
            
            if not models:
                st.warning("Please select at least one model to train")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, (name, model) in enumerate(models.items()):
                    status_text.text(f"Training {name}...")
                    model.fit(X_train, y_train)
                    results[name] = evaluate_model(model, X_test, y_test)
                    progress_bar.progress((i + 1) / len(models))
                
                status_text.text("Training complete!")
                
                # -------------------------------
                # Results Display
                # -------------------------------
                st.subheader("📈 Model Performance Comparison")
                
                # Convert results to DataFrame for easy display
                results_df = pd.DataFrame(results).T
                st.dataframe(results_df.style.highlight_max(axis=0))
                
                # Plot results
                fig, ax = plt.subplots(figsize=(10, 6))
                results_df.plot(kind='bar', ax=ax)
                plt.title("Model Performance Metrics")
                plt.ylabel("Score")
                plt.xlabel("Model")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Best model highlight
                best_model = results_df["Accuracy"].idxmax()
                st.success(f"Best model based on accuracy: {best_model} with {results_df.loc[best_model, 'Accuracy']:.4f} accuracy")


# In[ ]:




