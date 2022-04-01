# ExpSmoothing1
# нормализация
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
df = pd.read_excel("DatasetAAPl.xlsx")
scaler = MinMaxScaler()
df.iloc[:,1:] = scaler.fit_transform(df.iloc[:,1:])
df.to_excel("Dataset AAPl transformed.xlsx",index = 0)

