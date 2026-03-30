import pandas as pd

meta = pd.read_excel("metadata.xlsx")
print(meta.columns.tolist())
print(meta.head())
