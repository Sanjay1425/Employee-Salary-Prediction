import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv("employee_data.csv")

# Sidebar: show raw data
with st.sidebar:
    st.header("📂 Dataset Preview")
    if st.checkbox("Show Raw Data"):
        st.dataframe(df, use_container_width=True)


st.set_page_config(page_title="Employee Salary Predictor", layout="wide")
st.title("💼 Employee Salary Prediction App")

# Preprocess
X = df.drop("Salary", axis=1)
y = df["Salary"]

categorical_cols = ["Education", "Role", "Department", "Location"]
numerical_cols = ["Experience"]

preprocessor = ColumnTransformer([
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore")), categorical_cols),
    ("num", SimpleImputer(strategy="mean"), numerical_cols)
])

# Model pipeline
model = make_pipeline(
    preprocessor,
    XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
)

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

with st.expander("📊 Model Evaluation"):
    st.write(f"**Mean Squared Error (MSE):** `{mse:.2f}`")
    st.write(f"**R² Score:** `{r2:.4f}`")

# ==========================
# 🎯 Salary Prediction Form
# ==========================
st.subheader("🎯 Predict an Employee's Salary")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        education = st.selectbox("🎓 Education", df["Education"].unique(), key="edu")
        experience = st.slider("📊 Experience (Years)", 0, 50, 5, key="exp")

    with col2:
        role = st.selectbox("👨‍💻 Role", df["Role"].unique(), key="role")
        department = st.selectbox("🏢 Department", df["Department"].unique(), key="dept")

    with col3:
        location = st.selectbox("📍 Location", df["Location"].unique(), key="loc")
        st.markdown("")  # for spacing
        submitted = st.form_submit_button("🔍 Predict Salary", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([{
        "Education": education,
        "Experience": experience,
        "Role": role,
        "Department": department,
        "Location": location
    }])
    salary_prediction = model.predict(input_df)[0]
    st.success(f"💸 Estimated Salary: **${salary_prediction:,.2f}**")

# ==========================
# 📊 Data Visualizations
# ==========================
st.subheader("📊 Salary Insights")

# Chart 1: Salary Distribution
st.markdown("### 💰 Salary Distribution")
fig1, ax1 = plt.subplots(figsize=(5, 3))  # Compact chart
ax1.hist(df["Salary"], bins=10, color="#4F8EF7", edgecolor="black")
ax1.set_xlabel("Salary")
ax1.set_ylabel("Employees")
ax1.set_title("Salary Distribution", fontsize=10)
st.pyplot(fig1)

# Chart 2: Average Salary by Role
st.markdown("### 🧑‍💼 Average Salary by Role")
avg_salary_by_role = df.groupby("Role")["Salary"].mean().sort_values(ascending=False)

fig2, ax2 = plt.subplots(figsize=(5, 3))  # Compact chart
ax2.bar(avg_salary_by_role.index, avg_salary_by_role.values, color="#34A853")
ax2.set_ylabel("Avg Salary")
ax2.set_title("Average Salary by Role", fontsize=10)
ax2.tick_params(axis='x', rotation=45)
st.pyplot(fig2)
