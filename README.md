
# Introduction - InvestorsEye StockSight

Share prices are volatile & veru difficult to predict with a high degree of accuracy since there are many factors involved i.e. physical vs psychological, rational vs. irrational behaviour of human beings etc.

By looking at features like company announcements, revenue results etc ML can unearth patterns & insights which can increase the accuracy of predictions.

Stock market analysis is divided into two parts – Fundamental Analysis and Technical Analysis.

- Fundamental Analysis involves analyzing the company’s future profitability on the basis of its current business environment and financial performance.

- Technical Analysis, on the other hand, includes reading the charts and using statistical figures to identify the trends in the stock market.


# Problem Statement - Stocks prediction using LSTM

Long Short Term Memory (LSTM) for stocks prediction.

LSTMs are widely used for sequence prediction problems and have proven to be extremely effective. The reason they work so well is because it is able to store past information that is important, and forget the information that is not.

LSTM has three gates:

- The input gate: The input gate adds information to the cell state
- The forget gate: It removes the information that is no longer required by the model
- The output gate: Output Gate at LSTM selects the information to be shown as output

#Python #MachineLearning #Keras #DeepLearning #TensorFlow #Stocks

DataSet = Quandl - TATA Global Beverages, Microsoft



```python
#import packages
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 16,5

from matplotlib.lines import Line2D

#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

from datetime import datetime

vTicker = 'KEGN'


#read datafile
#df = pd.read_csv('data/NSE_KPLC_KenyaPower.csv')
df = pd.read_csv('data/NSE_KEGN_Kengen.csv')

#print the head
print(vTicker)
df.head()
```

    Using TensorFlow backend.
    

    KEGN
    




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Vol.</th>
      <th>Change %</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>08-05-2012</td>
      <td>8.65</td>
      <td>8.90</td>
      <td>8.50</td>
      <td>8.75</td>
      <td>186.70K</td>
      <td>1.16%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>09-05-2012</td>
      <td>8.75</td>
      <td>8.70</td>
      <td>8.50</td>
      <td>8.55</td>
      <td>94.90K</td>
      <td>-2.29%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10-05-2012</td>
      <td>8.55</td>
      <td>8.75</td>
      <td>8.45</td>
      <td>8.60</td>
      <td>76.80K</td>
      <td>0.58%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11-05-2012</td>
      <td>8.60</td>
      <td>8.85</td>
      <td>8.15</td>
      <td>8.60</td>
      <td>141.80K</td>
      <td>0.00%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14-05-2012</td>
      <td>8.60</td>
      <td>8.80</td>
      <td>8.60</td>
      <td>8.65</td>
      <td>91.70K</td>
      <td>0.58%</td>
    </tr>
  </tbody>
</table>
</div>



Variables – date, open, high, low, last, close, total_trade_quantity, and turnover.
Market is closed on weekends and public holidays.

Daily Closing price is the target variable to be predicted.



```python
#Format date to yyyymmdd and index data by date

df['Date'] = pd.to_datetime(df.Date,format='%d-%m-%Y')
df.index = df['Date']

#plot
plt.plot(df['Close'])
plt.title('[ ' + vTicker + ' ] - CLOSING PRICE TREND')
plt.show()
```


![png](output_4_0.png)


# Prediction Model - Long Short Term Memory (LSTM)

Create new dataFrame with only Date & Close price then then split it into training & validation sets to verify predictions.


```python
#Create new dataFrame with Date & Close price

data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

for i in range(0,len(data)):
     new_data['Date'][i] = data['Date'][i]
     new_data['Close'][i] = data['Close'][i]

new_data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012-05-08 00:00:00</td>
      <td>8.75</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2012-05-09 00:00:00</td>
      <td>8.55</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012-05-10 00:00:00</td>
      <td>8.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2012-05-11 00:00:00</td>
      <td>8.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012-05-14 00:00:00</td>
      <td>8.65</td>
    </tr>
  </tbody>
</table>
</div>



 Validation/control dataset = previous year’s data and Training dataset = next previous 4 years’ data before previous year.
 
 NOTE: Random splitting will destroy the date/time component.


```python
#Splitting training & validation datasets

#set date index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

#Split data for training - 80% and Validation 20%
vCount = round(len(dataset)*0.8)
#print(vCount)

train = dataset[0:vCount,:]
valid = dataset[vCount:,:]

dataset.shape, train.shape, valid.shape

#for i in range(5):
#    print(train[i])


```




    ((1776, 1), (1421, 1), (355, 1))




```python
#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Use previous 60 days data to predict
x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
    
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=2, batch_size=2, verbose=2)

#predicting 246 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)
```

    WARNING:tensorflow:From C:\ProgramData\Anaconda3\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py:475: DataConversionWarning: Data with input dtype object was converted to float64 by MinMaxScaler.
      warnings.warn(msg, DataConversionWarning)
    

    WARNING:tensorflow:From C:\ProgramData\Anaconda3\lib\site-packages\tensorflow\python\ops\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    Epoch 1/2
     - 23s - loss: 0.0034
    Epoch 2/2
     - 21s - loss: 0.0015
    


```python
rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
rms
```




    0.2390233086994984




```python
#for plotting
train = new_data[:vCount]
valid = new_data[vCount:]

valid['Predictions'] = closing_price
plt.plot(train['Close'], label='Closing Price Trend', lw=2)

#plt.plot(valid[['Close','Predictions']], dashes=[3, 2], label='Actual Vs. Predicted Closing Price', lw=2)
plt.plot(valid['Close'], label='Actual Closing Price', lw=1)
plt.plot(valid['Predictions'], dashes=[1, 2], label='Predicted Price', lw=2)

plt.title('[ ' + vTicker + ' ] - ACTUAL Vs. PREDICTED CLOSING PRICE').set_color('green')
plt.ylabel('Share Price (KES.)')
plt.xlabel('Trading Date')

plt.legend()
plt.show()

```

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """
    


![png](output_11_1.png)


# Inference

- LSTM model can be tuned for various parameters such as changing the number of LSTM layers, adding dropout value or increasing the number of epochs.

## Stock prices are affected by the news about the company and other factors like demonetization or merger/demerger of the companies. There are certain intangible factors as well which can often be impossible to predict beforehand.

