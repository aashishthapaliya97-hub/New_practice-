import streamlit as st
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt  
import numpy as np
import plotly.express as px

st.title("List of datas of peoples on the Titanic")
st.header("Header of the project")
st.subheader("Subheader of the project")        
st.write("This is a simple Streamlit app demonstrating titles, headers, and subheaders.")
st.text("This is some additional text to provide more context about the project.")
st.markdown("You can use **BOLD** to format your text easily *ITALIC bro*.")


df = sns.load_dataset("titanic")

selected_df = df[["embark_town","sex", "age", "class","fare","alive"]]
st.dataframe(df.head(5)) # this for interactive table
st.header("Only required datas for now")
st.dataframe(selected_df.head(5)) # this for interactive table
#st.table(df) # this for static table
st.metric(label="**Total number of passengers**", value=df.shape[0])

df= df.dropna(subset=["age"])
fig, ax = plt.subplots()
sns.kdeplot(data=df, x="age")   
st.pyplot(fig)


st.title("Tips Dataset Visualizations")
st.subheader("This data set shows how tips are related with total bills, gender and smoker vs non smoker .")

tips = sns.load_dataset("tips")

fig2= sns.relplot(data=tips, x="total_bill", y="tip", kind="scatter", hue="smoker")
plt.title("Scatter Plot: Relation between Total Bill and Tip")
st.pyplot(fig2)

fig3= sns.catplot(data=tips, x="day", y="total_bill", kind="box", hue="smoker")
plt.title("Box Plot: Total Bill Distribution by Day and Smoker Status")
st.pyplot(fig3)

