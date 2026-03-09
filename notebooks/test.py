import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/processed/dataset_master.csv')
df['label'] = df['label'].fillna(-1)
print(df)
print(df['label'].value_counts())
plt.figure(figsize=(6, 4))
plt.pie(df['label'].value_counts().values, autopct='%1.1f%%')
plt.title("Pie chart of label")
plt.legend(df['label'].value_counts().dropna().index, title="Label")
plt.show()