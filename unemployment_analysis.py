import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("Unemployment in India.csv")

# Remove unwanted spaces
df.columns = df.columns.str.strip()

print("="*60)
print("UNEMPLOYMENT ANALYSIS PROJECT")
print("="*60)



print("\nFirst 5 Records")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nColumns")
print(df.columns)

print("\nMissing Values")
print(df.isnull().sum())

# Remove missing values
df.dropna(inplace=True)

print("\nDataset Shape After Cleaning")
print(df.shape)


df['Date'] = pd.to_datetime(df['Date'])


print("\nStatistical Summary")
print(df.describe())


plt.figure(figsize=(18,12))



plt.subplot(3,2,1)

sns.lineplot(
    data=df,
    x='Date',
    y='Estimated Unemployment Rate (%)'
)

plt.title("Unemployment Trend Over Time")
plt.xticks(rotation=45)

plt.subplot(3,2,2)

top_states = df.groupby(
    'Region'
)['Estimated Unemployment Rate (%)'].mean()

top_states = top_states.sort_values(
    ascending=False
).head(10)

sns.barplot(
    x=top_states.values,
    y=top_states.index
)

plt.title("Top 10 States by Unemployment")

plt.subplot(3,2,3)

sns.histplot(
    df['Estimated Unemployment Rate (%)'],
    bins=20,
    kde=True,
    color='orange'
)

plt.title("Distribution of Unemployment Rate")

plt.subplot(3,2,4)

sns.boxplot(
    x=df['Estimated Unemployment Rate (%)']
)

plt.title("Outlier Detection")

plt.subplot(3,2,5)

sns.scatterplot(
    data=df,
    x='Estimated Labour Participation Rate (%)',
    y='Estimated Unemployment Rate (%)',
    hue='Area'
)

plt.title("Labour Participation vs Unemployment")

plt.subplot(3,2,6)

state_avg = df.groupby(
    'Region'
)['Estimated Unemployment Rate (%)'].mean()

state_avg = state_avg.sort_values(
    ascending=False
).head(15)

state_avg.plot(kind='bar')

plt.title("State-wise Average Unemployment")
plt.ylabel("Rate")

plt.tight_layout()
plt.show()


plt.figure(figsize=(10,6))

numeric_df = df.select_dtypes(
    include=np.number
)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()

covid = df[
    (df['Date'] >= '2020-03-01') &
    (df['Date'] <= '2020-12-31')
]

plt.figure(figsize=(12,6))

sns.lineplot(
    data=covid,
    x='Date',
    y='Estimated Unemployment Rate (%)'
)

plt.title("COVID-19 Impact on Unemployment")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


df['Month'] = df['Date'].dt.month

monthly = df.groupby(
    'Month'
)['Estimated Unemployment Rate (%)'].mean()

plt.figure(figsize=(10,5))

monthly.plot(
    marker='o',
    linewidth=3
)

plt.title("Monthly Average Unemployment")

plt.xlabel("Month")
plt.ylabel("Rate")

plt.grid(True)

plt.show()

print("\n")
print("="*60)
print("PROJECT INSIGHTS")
print("="*60)

print(
    "\nAverage Unemployment Rate:",
    round(
        df['Estimated Unemployment Rate (%)'].mean(),
        2
    )
)

print(
    "\nHighest Unemployment Rate:",
    df['Estimated Unemployment Rate (%)'].max()
)

print(
    "\nLowest Unemployment Rate:",
    df['Estimated Unemployment Rate (%)'].min()
)

highest_state = df.groupby(
    'Region'
)['Estimated Unemployment Rate (%)'].mean().idxmax()

print(
    "\nState with Highest Average Unemployment:"
)

print(highest_state)

print("\nAnalysis Completed Successfully")