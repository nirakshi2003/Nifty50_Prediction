 # Where's Nifty ?

## **Introduction :**

In this project, we will be predicting the high and low prices of Nifty50 index using a Stacked LSTM neural network. The model is trained on historical stock data to predict future prices. The LSTM model is chosen for its ability to handle time series data and capture temporal dependencies.

## **Reading the libraries:**

![image](https://github.com/nirakshi2003/Nifty50_Prediction/assets/96014974/46ab76ca-c6e7-4cc4-b7b7-846dc4b24375)

Pandas is a library that is used for data analysis. It also acts as a wrapper over Matplotlib and NumPy libraries. For instance, .plot() from Pandas performs a similar operation to Matplotlib plot operations.

NumPy is used for performing a wide variety of mathematical operations for arrays and matrices. In addition to this, the steps taken for computation from the NumPy library are effective and time-efficient.

Matplotlib is a low level graph plotting library in python that serves as a visualization utility. It creates publication quality plots and makes interactive figures that can zoom, pan, update.

## **Dataset:**

The dataset is used in this project contains historical Nifty stock prices. It includes various columns such as High, Low, Open, Close, Date, Shares Traded and Turnover.

#### Note: The data was taken from 
https://www.nseindia.com/reports-indices-historical-index-data

## **Data Preprocessing:**

Loading Data: The data is loaded using pandas.

Checking for Missing Values: There are no missing values and duplicate rows in our dataset. So there is no room for that error rectification.

![image](https://github.com/nirakshi2003/Nifty50_Prediction/assets/96014974/5a99072d-1ca5-4a8d-93aa-fc40080934aa)

Scaling Data: The High and Low columns are normalized using MinMaxScaler to bring values within the range of 0 to 1.

![image](https://github.com/nirakshi2003/Nifty50_Prediction/assets/96014974/1e50c00c-d22a-4a44-b57b-8ecd98f8a67d)


## **Data Exploration and Visualization:**  

We will visualize the High column (in y-axis) of the dataset by plotting it against days (in x-axis).

![image](https://github.com/nirakshi2003/Nifty50_Prediction/assets/96014974/6ade5d7b-0d35-4c07-af7f-6ddfc409e829)

Next up is the visualization of the Low column.

![image](https://github.com/nirakshi2003/Nifty50_Prediction/assets/96014974/3c332c91-283f-4dc2-93c9-64e1cfb456f6)



python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

# Scale High values
high_values = ds0["High"].values.reshape(-1, 1)
ds0["High"] = scaler.fit_transform(high_values)

# Scale Low values
low_values = ds0["Low"].values.reshape(-1, 1)
ds0["Low"] = scaler.fit_transform(low_values)

Model
An LSTM model is constructed using Keras. The model consists of three LSTM layers with dropout to prevent overfitting and two dense layers for the output.

python
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=(100, 1)))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

Training
The data is split into training and testing sets with a 70-30 ratio. The model is trained for 100 epochs with a batch size of 24.

python
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=24, verbose=1)

Evaluation
The model's performance is evaluated using Mean Squared Error (MSE). The predictions are plotted against the actual values to visualize the accuracy.

python
from sklearn.metrics import mean_squared_error
import math

train_predict = scaler.inverse_transform(model.predict(X_train))
test_predict = scaler.inverse_transform(model.predict(X_test))

train_score = math.sqrt(mean_squared_error(Y_train, train_predict))
test_score = math.sqrt(mean_squared_error(Y_test, test_predict))

print(f'Train Score: {train_score} RMSE')
print(f'Test Score: {test_score} RMSE')

Usage
To use this project, follow these steps:

Clone the repository.
Install the required dependencies.
Run the Jupyter notebook or script.

Requirements
Python 3.x
pandas
numpy
matplotlib
scikit-learn
keras
tensorflow
Install the required packages using:

bash
Copy code
pip install pandas numpy matplotlib scikit-learn keras tensorflow
