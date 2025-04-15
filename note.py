import pandas as pd

df = pd.read_csv("../data_processed/dataset_splitted.csv")
print(df['split'].value_counts())