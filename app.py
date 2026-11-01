import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="HR Analytics & Employee Performance", layout="wide")
st.title("HR Analytics & Employee Performance")
st.caption("Practical challenge • Synthetic employee data • EGP")

@st.cache_data
def load_data():
    return pd.read_csv("hr_employee_raw.csv")

raw = load_data()
df = raw.copy()
rows_before = len(df)
df = df.drop_duplicates()
df["Hire_Date"] = pd.to_datetime(df["Hire_Date"], errors="coerce")

numeric_cols = ["Age","Years_Experience","Monthly_Salary_EGP","Satisfaction_Score",
                "Performance_Score","Workload_Score","Training_Hours","Absence_Days","Overtime_Hours"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.loc[~df["Satisfaction_Score"].between(1,5),"Satisfaction_Score"] = np.nan
df.loc[~df["Performance_Score"].between(1,5),"Performance_Score"] = np.nan
df.loc[~df["Workload_Score"].between(1,5),"Workload_Score"] = np.nan
df.loc[df["Monthly_Salary_EGP"] <= 0,"Monthly_Salary_EGP"] = np.nan
df.loc[(df["Absence_Days"] < 0) | (df["Absence_Days"] > 60),"Absence_Days"] = np.nan
df.loc[(df["Training_Hours"] < 0) | (df["Training_Hours"] > 100),"Training_Hours"] = np.nan
df["Department"] = df["Department"].fillna("Unknown")
df["Monthly_Salary_EGP"] = df["Monthly_Salary_EGP"].fillna(
    df.groupby("Department")["Monthly_Salary_EGP"].transform("median"))
df["Satisfaction_Score"] = df["Satisfaction_Score"].fillna(df["Satisfaction_Score"].median())
df["Performance_Score"] = df["Performance_Score"].fillna(df["Performance_Score"].median())
df["Training_Hours"] = df["Training_Hours"].fillna(df["Training_Hours"].median())
df["Absence_Days"] = df["Absence_Days"].fillna(df["Absence_Days"].median())
df["Workload_Score"] = df["Workload_Score"].fillna(df["Workload_Score"].median())
df["Years_Experience"] = df["Years_Experience"].fillna(df["Years_Experience"].median())
df["Hire_Year"] = df["Hire_Date"].dt.year
df["Tenure_Years"] = (pd.Timestamp.today().normalize() - df["Hire_Date"]).dt.days / 365.25
df.loc[(df["Tenure_Years"] < 0) | (df["Tenure_Years"] > 50), "Tenure_Years"] = np.nan

st.sidebar.header("Filters")
departments = ["All"] + sorted(df["Department"].dropna().unique().tolist())
department = st.sidebar.selectbox("Department", departments)
statuses = ["All"] + sorted(df["Employee_Status"].dropna().unique().tolist())
status = st.sidebar.selectbox("Employee status", statuses)
modes = ["All"] + sorted(df["Work_Mode"].dropna().unique().tolist())
mode = st.sidebar.selectbox("Work mode", modes)
levels = ["All"] + sorted(df["Job_Level"].dropna().unique().tolist())
level = st.sidebar.selectbox("Job level", levels)

f = df.copy()
if department != "All": f = f[f["Department"] == department]
if status != "All": f = f[f["Employee_Status"] == status]
if mode != "All": f = f[f["Work_Mode"] == mode]
if level != "All": f = f[f["Job_Level"] == level]

active = f[f["Employee_Status"] == "Active"]
resigned_rate = f["Employee_Status"].eq("Resigned").mean()*100 if len(f) else 0
avg_salary = f["Monthly_Salary_EGP"].mean() if len(f) else 0
avg_perf = f["Performance_Score"].mean() if len(f) else 0
avg_sat = f["Satisfaction_Score"].mean() if len(f) else 0
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Employees", f"{len(f):,}")
c2.metric("Avg. monthly salary", f"{avg_salary:,.0f} EGP")
c3.metric("Avg. performance", f"{avg_perf:.2f}/5")
c4.metric("Avg. satisfaction", f"{avg_sat:.2f}/5")
c5.metric("Resigned share", f"{resigned_rate:.1f}%")

st.subheader("Department overview")
left,right = st.columns(2)
with left:
    dept_perf = f.groupby("Department",as_index=False).agg(
        Employees=("Employee_ID","nunique"), Avg_Performance=("Performance_Score","mean"))
    fig = px.bar(dept_perf,x="Department",y="Avg_Performance",color="Department",
                 title="Average performance by department",range_y=[0,5])
    st.plotly_chart(fig,use_container_width=True)
with right:
    dept_salary = f.groupby("Department",as_index=False).agg(
        Avg_Salary=("Monthly_Salary_EGP","mean"), Employees=("Employee_ID","nunique"))
    fig = px.bar(dept_salary,x="Department",y="Avg_Salary",color="Department",
                 title="Average monthly salary by department")
    st.plotly_chart(fig,use_container_width=True)

st.subheader("People factors")
left,right = st.columns(2)
with left:
    fig = px.scatter(f,x="Satisfaction_Score",y="Performance_Score",color="Department",
                     size="Training_Hours",hover_data=["Employee_ID","Workload_Score"],
                     title="Satisfaction vs performance")
    st.plotly_chart(fig,use_container_width=True)
with right:
    corr_cols = ["Monthly_Salary_EGP","Satisfaction_Score","Performance_Score",
                 "Workload_Score","Training_Hours","Absence_Days","Overtime_Hours"]
    corr = f[corr_cols].corr(numeric_only=True)
    fig,ax = plt.subplots(figsize=(7,5))
    sns.heatmap(corr,annot=True,cmap="coolwarm",center=0,fmt=".2f",ax=ax)
    ax.set_title("Correlation matrix")
    st.pyplot(fig,use_container_width=True)

st.subheader("Training and absence")
training = f.groupby("Department",as_index=False).agg(
    Avg_Training_Hours=("Training_Hours","mean"),
    Avg_Performance=("Performance_Score","mean"),
    Avg_Absence_Days=("Absence_Days","mean"))
fig = px.scatter(training,x="Avg_Training_Hours",y="Avg_Performance",
                 size="Avg_Absence_Days",color="Department",hover_name="Department",
                 title="Department training vs performance")
st.plotly_chart(fig,use_container_width=True)

with st.expander("Data quality checks"):
    st.write(f"Rows before duplicate removal: {rows_before:,}")
    st.write(f"Rows after duplicate removal: {len(df):,}")
    st.dataframe(df.isna().sum().rename("Missing values").to_frame())
    st.caption("This is synthetic training data. Imputation choices are teaching examples; HR decisions must not be based on imputed records without validation.")

with st.expander("Explore filtered employee records"):
    st.dataframe(f,use_container_width=True)
    st.download_button("Download filtered data",f.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_hr_data.csv",mime="text/csv")

st.subheader("Suggested HR actions")
st.markdown("""
- Investigate departments with low average performance using role-specific context before proposing interventions.
- Review satisfaction and workload together; correlation does not establish causation.
- Evaluate training programs with a before/after design rather than assuming more training causes higher performance.
- Compare salary only within comparable roles and job levels; department averages can be misleading.
- Treat attrition metrics as a planning signal, not as a basis for individual employment decisions.
""")
