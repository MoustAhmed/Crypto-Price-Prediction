import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Function to fetch historical cryptocurrency data using the cryptocompare API
def fetch_crypto_data(symbol, currency, limit, api_key):
    url = f'https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym={currency}&limit={limit}&api_key={api_key}'
    response = requests.get(url)
    data = response.json()['Data']['Data']
    
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    return df

# Replace 'YOUR_API_KEY' with your actual API key from cryptocompare
api_key = 'YOUR_API_KEY'
symbol = 'BTC'
currency = 'USD'
limit = 100
crypto_data = fetch_crypto_data(symbol, currency, limit, api_key)

# Feature engineering
crypto_data['MA10'] = crypto_data['close'].rolling(window=10).mean()
crypto_data['MA50'] = crypto_data['close'].rolling(window=50).mean()
crypto_data['EMA'] = crypto_data['close'].ewm(span=20, adjust=False).mean()

# Create target variable (shift close prices to predict the next day)
crypto_data['Target'] = crypto_data['close'].shift(-1)

# Drop NaN values
crypto_data = crypto_data.dropna()

# Split the data into training and testing sets
X = crypto_data[['open', 'high', 'low', 'close', 'volumefrom', 'volumeto', 'MA10', 'MA50', 'EMA']]
y = crypto_data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')

# Visualize predictions
plt.figure(figsize=(12, 6))
plt.plot(crypto_data.index[-len(predictions):], predictions, label='Predicted')
plt.plot(crypto_data.index[-len(y_test):], y_test, label='Actual', alpha=0.7)
plt.legend()
plt.title('Cryptocurrency Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()
