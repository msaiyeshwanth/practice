# practice


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import shap


# Load your dataset
# df = pd.read_csv('your_data.csv')  # Uncomment this line to load your actual dataset

# Assuming the data is loaded in df
# Example creation of the DataFrame (remove when using actual dataset)
np.random.seed(0)
date_rng = pd.date_range(start='2020-01-01', end='2020-06-01', freq='T')
df = pd.DataFrame(date_rng, columns=['datetime'])
df = df.assign(**{f'feature{i}': np.random.randn(len(df)) for i in range(1, 32)})
df['target'] = np.random.randn(len(df))

# Convert datetime column to datetime object if necessary
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# Separate features and target
features = df.drop(columns=['target'])
target = df['target']

# Normalize the data
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

features_scaled = scaler_features.fit_transform(features)
target_scaled = scaler_target.fit_transform(target.values.reshape(-1, 1))

# Create a new DataFrame for scaled features
features_scaled_df = pd.DataFrame(features_scaled, index=features.index, columns=features.columns)
features_scaled_df['target'] = target_scaled

# Function to create sequences
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data.iloc[i:i + seq_length].values)
        labels.append(data.iloc[i + seq_length]['target'])
    return np.array(sequences), np.array(labels)

# Using a long sequence length to capture long-term dependencies
seq_length = 1440  # Example: using past 1440 minutes (24 hours) to predict the next value
X, y = create_sequences(features_scaled_df, seq_length)

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape the data for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))






model = Sequential()
# First LSTM layer with 100 units, returns full sequence
model.add(LSTM(100, return_sequences=True, input_shape=(seq_length, X_train.shape[2])))
# Second LSTM layer with 50 units, returns only last output
model.add(LSTM(50))
# Dense layer to output the final prediction
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model on the training data
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)





y_pred = model.predict(X_test)

# Invert the scaling for the target
y_test_scaled = scaler_target.inverse_transform(y_test.reshape(-1, 1))
y_pred_scaled = scaler_target.inverse_transform(y_pred)

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(y_test_scaled, label='Actual')
plt.plot(y_pred_scaled, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('LSTM Time Series Prediction')
plt.legend()
plt.show()





# Use a subset of training data for the explainer due to computational cost
explainer = shap.KernelExplainer(model.predict, X_train[:100])
# Using a subset of test data for SHAP values
shap_values = explainer.shap_values(X_test[:10], nsamples=100)

# Plot SHAP values for a specific instance
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0][0], features_scaled_df.columns)
