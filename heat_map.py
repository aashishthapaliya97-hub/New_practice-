import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Seaborn Charts Demo")

# Load sample dataset
df = sns.load_dataset('tips')

# Chart 1: Histogram
st.header("1. Distribution of Total Bill")
fig1, ax1 = plt.subplots()
sns.histplot(df['total_bill'], bins=20, ax=ax1)
st.pyplot(fig1)

# Chart 2: Scatter Plot
st.header("2. Tip vs Total Bill")
fig2, ax2 = plt.subplots()
sns.scatterplot(data=df, x='total_bill', y='tip', ax=ax2)
st.pyplot(fig2)

# Chart 3: Box Plot
st.header("3. Tips by Day")
fig3, ax3 = plt.subplots()
sns.boxplot(data=df, x='day', y='tip', ax=ax3)
st.pyplot(fig3)

# Chart 4: Correlation Heatmap
st.header("4. Correlation Matrix")
fig4, ax4 = plt.subplots()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax4)
st.pyplot(fig4)
