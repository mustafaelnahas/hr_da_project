import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# 1) Import and inspect
raw = pd.read_csv("hr_employee_raw.csv")
print(raw.shape)
print(raw.head())
print(raw.info())
print(raw.isna().sum())
print("Exact duplicates:", raw.duplicated().sum())

# 2) NumPy: robust summaries
salary = pd.to_numeric(raw["Monthly_Salary_EGP"],errors="coerce").dropna().to_numpy()
print("Salary median:", np.median(salary))
print("Salary 90th percentile:", np.percentile(salary,90))

# 3) Clean
df = raw.copy()
df["Hire_Date"] = pd.to_datetime(df["Hire_Date"],errors="coerce")
df = df.drop_duplicates()
for col in ["Age","Years_Experience","Monthly_Salary_EGP","Satisfaction_Score",
            "Performance_Score","Workload_Score","Training_Hours","Absence_Days","Overtime_Hours"]:
    df[col] = pd.to_numeric(df[col],errors="coerce")
for col in ["Satisfaction_Score","Performance_Score","Workload_Score"]:
    df.loc[~df[col].between(1,5),col] = np.nan
df.loc[df["Monthly_Salary_EGP"]<=0,"Monthly_Salary_EGP"] = np.nan
df.loc[(df["Absence_Days"]<0)|(df["Absence_Days"]>60),"Absence_Days"] = np.nan
df["Department"] = df["Department"].fillna("Unknown")
df["Monthly_Salary_EGP"] = df["Monthly_Salary_EGP"].fillna(
    df.groupby("Department")["Monthly_Salary_EGP"].transform("median"))
df["Satisfaction_Score"] = df["Satisfaction_Score"].fillna(df["Satisfaction_Score"].median())
df["Performance_Score"] = df["Performance_Score"].fillna(df["Performance_Score"].median())
df["Training_Hours"] = df["Training_Hours"].fillna(df["Training_Hours"].median())
df["Absence_Days"] = df["Absence_Days"].fillna(df["Absence_Days"].median())

# 4) Analysis
print(df.groupby("Department").agg(
    Employees=("Employee_ID","nunique"),
    Avg_Salary=("Monthly_Salary_EGP","mean"),
    Avg_Performance=("Performance_Score","mean"),
    Avg_Satisfaction=("Satisfaction_Score","mean"),
    Avg_Absence=("Absence_Days","mean")
).round(2))
print(df[["Monthly_Salary_EGP","Satisfaction_Score","Performance_Score",
          "Workload_Score","Training_Hours","Absence_Days"]].corr().round(2))

# 5) Matplotlib
df.groupby("Department")["Performance_Score"].mean().sort_values().plot(
    kind="barh",title="Average performance by department")
plt.xlabel("Performance score (1–5)")
plt.tight_layout()
plt.show()

# 6) Seaborn
sns.boxplot(data=df,x="Department",y="Monthly_Salary_EGP")
plt.xticks(rotation=25)
plt.title("Salary distribution by department")
plt.tight_layout()
plt.show()

# 7) Plotly
summary = df.groupby("Department",as_index=False).agg(
    Avg_Satisfaction=("Satisfaction_Score","mean"),
    Avg_Performance=("Performance_Score","mean"))
fig = px.scatter(summary,x="Avg_Satisfaction",y="Avg_Performance",
                 color="Department",hover_name="Department",
                 title="Department satisfaction vs performance")
fig.show()
