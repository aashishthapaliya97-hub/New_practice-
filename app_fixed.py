import streamlit as st
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  # needed for regression line

# Load dataset
tips = sns.load_dataset("tips")

# Streamlit App Title
st.title("Tips Dataset Interactive Visualization")
st.header("Data of the tips project")
st.dataframe(tips)

# -------------------------------
# Bar Charts
# -------------------------------
st.title("Tips Dataset Bar Charts")

# Bar Chart 1: Total Bill per Day
st.header("Total Bill per Day")
total_bill_day = tips.groupby("day")["total_bill"].sum().reset_index()
fig_total_bill = px.bar(
    total_bill_day,
    x="day",
    y="total_bill",
    color="day",
    title="Total Bill per Day",
    text="total_bill"
)
fig_total_bill.update_traces(texttemplate='%{text:.2f}', textposition='outside')
st.plotly_chart(fig_total_bill, use_container_width=True)

# Bar Chart 2: Number of Customers per Day
st.header("Number of Customers per Day")
count_day = tips.groupby("day").size().reset_index(name="count")
fig_count = px.bar(
    count_day,
    x="day",
    y="count",
    color="day",
    title="Number of Records per Day",
    text="count"
)
fig_count.update_traces(textposition='outside')
st.plotly_chart(fig_count, use_container_width=True)

# Average bill per day table
avg_bill_per_day = tips.groupby('day')['total_bill'].mean().reset_index()
avg_bill_per_day.rename(columns={'total_bill': 'avg_total_bill'}, inplace=True)
st.subheader("Average Total Bill Per Day")
st.dataframe(avg_bill_per_day)

# -------------------------------
# Scatter Plots
# -------------------------------
st.header("Section 1: Scatter Plots (Tips vs Total Bill)")

# Scatter Plot 1: Tips vs Total Bill
st.subheader("Scatter Plot 1: Tips vs Total Bill")
fig1 = px.scatter(
    tips,
    x="total_bill",
    y="tip",
    title="Tips vs Total Bill",
    hover_data=["day", "time", "size"]
)
st.plotly_chart(fig1, use_container_width=True)

# Scatter Plot 1 with regression line (all data)
slope, intercept = np.polyfit(tips["total_bill"], tips["tip"], 1)
tips["reg_line"] = slope * tips["total_bill"] + intercept
st.subheader("Scatter Plot: Tips vs Total Bill (with Regression Line)")
fig1_line = px.scatter(
    tips,
    x="total_bill",
    y="tip",
    color="tip",
    title="Tips vs Total Bill (with Regression Line)",
    hover_data=["day", "time", "size"],
    color_continuous_scale="Viridis"
)
fig1_line.add_traces(px.line(tips, x="total_bill", y="reg_line").data)
st.plotly_chart(fig1_line, use_container_width=True)
corr = tips["total_bill"].corr(tips["tip"])
st.write(f"**Correlation (r): {corr:.3f}**")
st.write("**This scatter plot shows a moderate positive correlation between total bill and tip.**")

# Scatter Plot 1: Indexed by Sex
st.subheader("Scatter Plot 1: Tips vs Total Bill (Indexed by Sex)")
fig_sex = px.scatter(
    tips,
    x="total_bill",
    y="tip",
    color="sex",
    title="Tips vs Total Bill (Indexed by Sex)",
    hover_data=["day", "time", "size"]
)
st.plotly_chart(fig_sex, use_container_width=True)

# -------------------------------
# Bar Graphs and Average Table by Sex
# -------------------------------
st.header("Bar Graph 1: Total Bill by Sex")
bill_by_sex = tips.groupby("sex")["total_bill"].sum().reset_index()
fig_bar1 = px.bar(
    bill_by_sex,
    x="sex",
    y="total_bill",
    title="Total Bill Sum by Sex",
    text="total_bill",
    color="sex"
)
fig_bar1.update_traces(textposition="outside")
st.plotly_chart(fig_bar1, use_container_width=True)

st.header("Bar Graph 2: Number of Customers by Sex")
count_by_sex = tips["sex"].value_counts().reset_index()
count_by_sex.columns = ["sex", "count"]
fig_bar2 = px.bar(
    count_by_sex,
    x="sex",
    y="count",
    title="Number of Customers by Sex",
    text="count",
    color="sex"
)
fig_bar2.update_traces(textposition="outside")
st.plotly_chart(fig_bar2, use_container_width=True)

st.header("Average Total Bill by Sex")
avg_by_sex = tips.groupby("sex")["total_bill"].mean().reset_index()
avg_by_sex.columns = ["Sex", "Average Total Bill"]
st.table(avg_by_sex)

# Separate Tables by Sex
# -------------------------------
st.header("Tips Summary: Male Only")
male = tips[tips["sex"] == "Male"]
male_summary = male.groupby("day")["tip"].agg(['mean', 'median']).reset_index()
male_summary.columns = ["Day", "Mean Tip", "Median Tip"]
st.table(male_summary)

st.header("Tips Summary: Female Only")
female = tips[tips["sex"] == "Female"]
female_summary = female.groupby("day")["tip"].agg(['mean', 'median']).reset_index()
female_summary.columns = ["Day", "Mean Tip", "Median Tip"]
st.table(female_summary)

# Total Bill Summary Tables by gender Mean and Median
st.header("Total Bill Summary: Male Only")
male = tips[tips["sex"] == "Male"]
male_bill_summary = male.groupby("day")["total_bill"].agg(['mean', 'median']).reset_index()
male_bill_summary.columns = ["Day", "Mean Total Bill", "Median Total Bill"]
st.table(male_bill_summary)

st.header("Total Bill Summary: Female Only")
female = tips[tips["sex"] == "Female"]
female_bill_summary = female.groupby("day")["total_bill"].agg(['mean', 'median']).reset_index()
female_bill_summary.columns = ["Day", "Mean Total Bill", "Median Total Bill"]
st.table(female_bill_summary)


# -------------------------------
# Scatter Plot 2: Indexed by Smoker
# -------------------------------
st.subheader("Scatter Plot 2: Tips vs Total Bill (Indexed by Smoker)")
fig_smoker = px.scatter(
    tips,
    x="total_bill",
    y="tip",
    color="smoker",
    title="Tips vs Total Bill (Indexed by Smoker)",
    hover_data=["day", "time", "size"]
)
st.plotly_chart(fig_smoker, use_container_width=True)

# -------------------------------
# Scatter Plots with Regression Line (No statsmodels)
# -------------------------------
# Smokers Only
st.subheader("Scatter Plot (Smokers Only)")
smokers = tips[tips["smoker"] == "Yes"]
slope_s, intercept_s = np.polyfit(smokers["total_bill"], smokers["tip"], 1)
smokers["predicted_tip"] = slope_s * smokers["total_bill"] + intercept_s
corr_smoker = smokers["total_bill"].corr(smokers["tip"])

fig_smoker_only = px.scatter(
    smokers,
    x="total_bill",
    y="tip",
    title="Tips vs Total Bill (Smokers Only)",
    color="tip"
)
fig_smoker_only.add_traces(px.line(smokers, x="total_bill", y="predicted_tip").data)
st.plotly_chart(fig_smoker_only, use_container_width=True)
st.write(f"📌 **Correlation (Smokers):** `{corr_smoker:.3f}`")
st.write(f"📉 Regression Line: **tip = {slope_s:.3f} × total_bill + {intercept_s:.3f}**")

# Non-Smokers Only
st.subheader("Scatter Plot (Non-Smokers Only)")
nonsmokers = tips[tips["smoker"] == "No"]
slope_ns, intercept_ns = np.polyfit(nonsmokers["total_bill"], nonsmokers["tip"], 1)
nonsmokers["predicted_tip"] = slope_ns * nonsmokers["total_bill"] + intercept_ns
corr_nonsmoker = nonsmokers["total_bill"].corr(nonsmokers["tip"])

fig_nonsmoker_only = px.scatter(
    nonsmokers,
    x="total_bill",
    y="tip",
    title="Tips vs Total Bill (Non-Smokers Only)",
    color="tip"
)
fig_nonsmoker_only.add_traces(px.line(nonsmokers, x="total_bill", y="predicted_tip").data)
st.plotly_chart(fig_nonsmoker_only, use_container_width=True)
st.write(f"📌 **Correlation (Non-Smokers):** `{corr_nonsmoker:.3f}`")
st.write(f"📉 Regression Line: **tip = {slope_ns:.3f} × total_bill + {intercept_ns:.3f}**")

# Total Bill Summary Tables by Smoker Status Mean and Median
st.header("Total Bill Summary: Smokers Only")
smokers = tips[tips["smoker"] == "Yes"]
smokers_bill_summary = smokers.groupby("day")["total_bill"].agg(['mean', 'median']).reset_index()
smokers_bill_summary.columns = ["Day", "Mean Total Bill", "Median Total Bill"]
st.table(smokers_bill_summary)

# Total Bill Summary: Non-Smokers Only  mean and median
st.header("Total Bill Summary: Non-Smokers Only")
nonsmokers = tips[tips["smoker"] == "No"]
nonsmokers_bill_summary = nonsmokers.groupby("day")["total_bill"].agg(['mean', 'median']).reset_index()
nonsmokers_bill_summary.columns = ["Day", "Mean Total Bill", "Median Total Bill"]
st.table(nonsmokers_bill_summary)

# -------------------------------
# Box Plots
# -------------------------------


# Mean and median (Smoker vs non smoker tips)

st.header("Tips Summary: Smokers Only")
smokers = tips[tips["smoker"] == "Yes"]
smokers_summary = smokers.groupby("day")["tip"].agg(['mean', 'median']).reset_index()
smokers_summary.columns = ["Day", "Mean Tip", "Median Tip"]
st.table(smokers_summary)

st.header("Tips Summary: Non-Smokers Only")
nonsmokers = tips[tips["smoker"] == "No"]
nonsmokers_summary = nonsmokers.groupby("day")["tip"].agg(['mean', 'median']).reset_index()
nonsmokers_summary.columns = ["Day", "Mean Tip", "Median Tip"]
st.table(nonsmokers_summary)



# Box Plot 1: Indexed by Sex
st.subheader("Box Plot 1: Tips vs Day (Indexed by Sex)")
fig_box_sex = px.box(
    tips,
    x="day",
    y="tip",
    color="sex",
    title="Tip Distribution Across Days (Colored by Sex)"
)
st.plotly_chart(fig_box_sex, use_container_width=True)

# Box Plot 2: Indexed by Smoker
st.subheader("Box Plot 2: Tips vs Day (Indexed by Smoker)")
fig_box_smoker = px.box(
    tips,
    x="day",
    y="tip",
    color="smoker",
    title="Tip Distribution Across Days (Colored by Smoker)"
)
st.plotly_chart(fig_box_smoker, use_container_width=True)
