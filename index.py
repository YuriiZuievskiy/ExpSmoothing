import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

bestAlpha = 0.8

def exponential_smoothing(series, alpha):
  result = [series[0]]
  for n in range(1, len(series)):
    result.append(alpha * series[n] + (1 - alpha) * result[n-1])

  return result

def augment_series(series):
  smoothed = exponential_smoothing(series, bestAlpha)
  result = []
  for idx, val in enumerate(series):
    result.append(val)
    if idx + 1 < len(smoothed):
      result.append(smoothed[idx + 1])
      
  return result

df = pd.read_excel("DatasetAAPl.xlsx")

# рисует график исходных данных и сглаженных
# with plt.style.context('seaborn-white'):
#   plt.figure(figsize=(20, 8))
#   plt.plot(exponential_smoothing(df.iloc[:, 1], bestAlpha), label="Alpha {}".format(bestAlpha))
#   plt.plot(df.iloc[:, 1].values, "c", label = "Actual")
#   plt.legend(loc="best")
#   plt.axis('tight')
#   plt.title("Exponential Smoothing")
#   plt.grid(True)
#   plt.show()

# расширение первых четырех колонок датафрейма
augmented = []
for col in range(1, 5):
  augmented.append(augment_series(df.iloc[:, col]))

augmentedDataframe = pd.DataFrame(data=augmented).T
augmentedDataframe.to_excel("Dataset_augmented.xlsx",index = 0)

scaler = MinMaxScaler()
augmentedDataframe.iloc[:,0:] = scaler.fit_transform(augmentedDataframe.iloc[:,0:])
augmentedDataframe.to_excel("Dataset AAPl transformed.xlsx",index = 0)
