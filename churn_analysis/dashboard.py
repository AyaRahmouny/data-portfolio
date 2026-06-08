# ============================================
# CUSTOMER CHURN ANALYSIS DASHBOARD
# ============================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Customer Churn Analysis",
    page_icon="📉",
    layout="wide"
)

# ---- LOAD DATA & MODEL ----
@st.cache_data
def load_data():
    return pd.read_csv('data/clean_dataset.csv')

@st.cache_resource
def load_model():
    return joblib.load('models/logistic_model.pkl')

df = load_data()
model = load_model()

# ---- COLORS ----
BLUE = '#185FA5'
LIGHT_BLUE = '#85B7EB'
RED = '#E24B4A'
GRAY = '#888780'

# ---- TITLE ----
st.title("📉 Customer Churn Analysis")
st.markdown("Analysis of **7,043 telecom customers** — identifying churn patterns and retention strategies")
st.divider()

# ---- SECTION 1: KPIs ----
st.header("📈 Overview")

total = len(df)
churned = len(df[df['Churn'] == 'Yes'])
churn_rate = round(churned / total * 100, 1)
avg_charges = round(df['MonthlyCharges'].mean(), 2)
senior_churn = round(len(df[(df['SeniorCitizen']==1) & (df['Churn']=='Yes')]) / len(df[df['SeniorCitizen']==1]) * 100, 1)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{total:,}")
col2.metric("Churned", f"{churned:,}", f"{churn_rate}%")
col3.metric("Avg Monthly Charges", f"${avg_charges}")
col4.metric("Senior Churn Rate", f"{senior_churn}%")

st.divider()

# ---- SECTION 2: TOP 3 CHARTS ----
st.header("🔍 Key Insights")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Churn by Contract")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x='Contract', hue='Churn', data=df,
                  palette={'No': BLUE, 'Yes': RED}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Customers")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(title='Churn')
    st.pyplot(fig)

with col2:
    st.subheader("Churn by Tenure Group")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x='tenure_group', hue='Churn', data=df,
                  palette={'No': BLUE, 'Yes': RED}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Customers")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(title='Churn')
    st.pyplot(fig)

with col3:
    st.subheader("Monthly Charges vs Churn")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df,
                palette={'No': BLUE, 'Yes': RED}, ax=ax)
    ax.set_xlabel("Churn")
    ax.set_ylabel("Monthly Charges ($)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)

st.divider()

# ---- SECTION 3: EXPLORE DATA ----
st.header("🔎 Explore Customers")

col1, col2 = st.columns(2)
with col1:
    churn_filter = st.selectbox("Filter by Churn", ["All", "Yes", "No"])
with col2:
    contract_filter = st.selectbox("Filter by Contract", ["All"] + df['Contract'].unique().tolist())

filtered_df = df.copy()
if churn_filter != "All":
    filtered_df = filtered_df[filtered_df['Churn'] == churn_filter]
if contract_filter != "All":
    filtered_df = filtered_df[filtered_df['Contract'] == contract_filter]

st.dataframe(
    filtered_df[['gender', 'tenure', 'Contract',
                 'MonthlyCharges', 'num_services',
                 'tenure_group', 'Churn']].head(50),
    use_container_width=True
)

col1, col2 = st.columns(2)
with col1:
    st.caption(f"Showing {len(filtered_df):,} customers")
with col2:
    st.download_button(
        label="📥 Download filtered data",
        data=filtered_df.to_csv(index=False),
        file_name="filtered_customers.csv",
        mime="text/csv"
    )

st.divider()

# ---- SECTION 4: CHURN PREDICTION ----
st.header("🔮 Churn Prediction")
st.markdown("Enter the key customer details to predict churn risk")

col1, col2 = st.columns(2)

with col1:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 0.0, 120.0, 50.0)

with col2:
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

if st.button("🔮 Predict", use_container_width=True):

    final_ready = pd.read_csv('data/final_ready.csv')
    sample = final_ready.iloc[0].copy()

    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
    security_map = {"No": 0, "No internet service": 1, "Yes": 2}

    sample['Contract'] = contract_map[contract]
    sample['tenure'] = tenure
    sample['MonthlyCharges'] = monthly_charges
    sample['InternetService'] = internet_map[internet_service]
    sample['OnlineSecurity'] = security_map[online_security]
    sample['TechSupport'] = security_map[tech_support]

    if tenure <= 12: sample['tenure_group'] = 0
    elif tenure <= 24: sample['tenure_group'] = 1
    elif tenure <= 48: sample['tenure_group'] = 2
    else: sample['tenure_group'] = 3

    sample['high_value'] = int(monthly_charges > df['MonthlyCharges'].median())
    sample['avg_monthly_spend'] = monthly_charges

    input_df = pd.DataFrame([sample]).drop('Churn', axis=1)

    prediction = model.predict(input_df)[0]
    probability = round(model.predict_proba(input_df)[0][1] * 100, 1)

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ High churn risk — Probability: {probability}%")
        st.markdown("**Recommended actions:**")
        st.markdown("- Offer a long-term contract discount")
        st.markdown("- Assign dedicated support")
        st.markdown("- Provide loyalty incentives")
    else:
        st.success(f"✅ Low churn risk — Churn Probability: {probability}%")
        st.markdown("**Recommended actions:**")
        st.markdown("- Continue current strategy")
        st.markdown("- Consider upselling services")