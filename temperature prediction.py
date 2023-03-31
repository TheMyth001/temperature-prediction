import math
import scipy
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error


def calculate_quantile (column, data_table):
    Q1 = data_table[[column]].quantile(0.25)[0]
    Q3 = data_table[[column]].quantile(0.75)[0]
    IQR = Q3 - Q1
    min = data_table[[column]].min()[0]
    max = data_table[[column]].max()[0]
    min_IQR = Q1 - 1.5*IQR
    max_IQR = Q3 + 1.5*IQR
    return Q1, Q3, min, max, min_IQR, max_IQR


def combine_date(data_table):
    list_tab = []
    for i in range(data_table.shape[0]):
        list_tab.append(f'{data_table.loc[i, "Date"]}T{data_table.loc[i, "Time"].split(":")[0].zfill(2)}')
    return np.array(list_tab, dtype="datetime64")


# converting date and time columns to single date-time column
df = pd.read_csv("air_quality.csv")
df["DateTime"] = combine_date(df)

# convert to hourly data
df = df[["DateTime", "O3", "CO", "NO2", "SO2", "NO", "CO2", "VOC", "PM1",
         "PM2.5", "PM4", "PM10", "TSP", "TEMP", "HUM", "WS", "WD", "ISPU"]]
df2 = df.groupby(["DateTime"]).mean()

# delete first and last rows to avoid missing value extrapolation
df2.drop(index=[df2.index[0], df2.index[df2.shape[0]-1]], inplace=True)

# removing outliers and replacing using extrapolation
for i in df2.columns:
    Q1, Q3, min, max, min_IQR, max_IQR = calculate_quantile(i, df2)


    def outliers_to_nan(x, min_IQR=min_IQR, max_IQR=max_IQR):
        if x > max_IQR or x < min_IQR:
            x = np.nan
        else:
            x = x
        return x


    def outliers_to_nan_humidity(x, min_IQR=min_IQR, max_IQR=100):
        if x > max_IQR or x < min_IQR:
            x = np.nan
        else:
            x = x
        return x


    if i == "HUM":
        df2[i] = df2[i].map(outliers_to_nan_humidity)
        df2[i] = df2[i].interpolate(method="linear")
    else:
        df2[i] = df2[i].map(outliers_to_nan)
        df2[i] = df2[i].interpolate(method="linear")

# dealing with positive skew in temperature distribution
dataset = np.log1p(df2[["TEMP"]].values)
distribution_df = pd.DataFrame({"Temp": df2[["TEMP"]].values.T[0], "Log(Temp)": dataset.T[0]})
plt.figure(figsize=(12, 5))
distribution_df.hist()

log1ptminus2 = []
log1ptminus1 = []
log1pt = []
hour = []
for i in range(2, len(df2)):
    log1ptminus2.append(np.log1p(df2["TEMP"][i-2]))
    log1ptminus1.append(np.log1p(df2["TEMP"][i-1]))
    log1pt.append(np.log1p(df2["TEMP"][i]))
    hour.append(int(str(df2.index[i])[11:13]))

df2.drop(index=[df2.index[0], df2.index[1]], inplace=True)
df2["log1ptminus2"] = np.array(log1ptminus2)
df2["log1ptminus1"] = np.array(log1ptminus1)
df2["Hour"] = np.array(hour)
df2["log1pt"] = np.array(log1pt)

# seperating into input and output
input_data = df2.drop("log1pt", axis=1)
output_data = df2["log1pt"]

# splitting the data into train and test sets
train_size = int(len(df2) * 0.75)
test_size = len(df2) - train_size
trainX, testX = np.array(input_data[0:train_size]), np.array(input_data[train_size:len(df2)])
trainY, testY = np.array(output_data[0:train_size]), np.array(output_data[train_size:len(df2)])


lookback = 1
# create model
model = keras.Sequential()
model.add(keras.layers.LSTM(5, input_shape=(20, lookback)))
model.add(keras.layers.Dense(1))

model.compile(
    loss="mean_squared_error",
    optimizer="adam"
)

model.fit(trainX, trainY, epochs=300, batch_size=32, verbose=2)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = np.expm1(trainPredict)
trainY = np.expm1(trainY)
testPredict = np.expm1(testPredict)
testY = np.expm1(testY)

trainError = math.sqrt(mean_squared_error(trainY, trainPredict[:, 0]))
testError = math.sqrt(mean_squared_error(testY, testPredict[:, 0]))

print(f"Train error: {trainError:0.2f} \t Train std dev: {np.std(trainY):0.2f}")
print(f"Test error: {testError:0.2f} \t Test std dev: {np.std(testY):0.2f}")
