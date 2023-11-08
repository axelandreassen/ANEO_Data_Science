import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Seed for reproducibility
np.random.seed(0)

# Generate 'Marketing' and 'Visitors' time series data
data_length = 1000
marketing = np.random.randn(data_length).cumsum() + 50
visitors = np.random.randn(data_length).cumsum() + 200

# Generate 'Sales' based on non-linear relationships with lagged 'Marketing' and 'Visitors'
sales_lag = 5  # Define the lag value
sales = np.zeros(data_length)

for i in range(sales_lag, data_length):
    # Non-linear combination of lagged values
    sales[i] = np.sin(marketing[i - sales_lag]) + np.sqrt(abs(visitors[i - sales_lag])) + np.random.randn()

# Ensure no zero values
sales = sales + abs(min(sales)) + 1

data = pd.DataFrame({
    'Sales': sales,
    'Marketing': marketing,
    'Visitors': visitors
})


# Function to create dataset with lags for multiple variables
def create_multivariate_dataset(data, min_lag=1, max_lag=3):
    X, y = [], []
    for i in range(max_lag, len(data)):
        lagged_data = data.iloc[i-max_lag:i-min_lag+1, 1:]  # Shape: [time_steps, num_predictors]
        X.append(lagged_data.values)
        y.append(data.iloc[i, 0])  # Target variable (Sales)
    return np.array(X), np.array(y)

# Hyperparameter ranges
min_lag_values = [1]
max_lag_values = [10]
lstm_units_values = [50]

# k-Fold for validation
k = 5
kf = KFold(n_splits=k, shuffle=False)

# Grid Search
best_mape = float('inf')
best_hyperparams = None

print("Min Max Units || MAPE")
for min_lag in min_lag_values:
    for max_lag in max_lag_values:
        for lstm_units in lstm_units_values:
            if max_lag > min_lag:
                print(f"{min_lag}   {max_lag}   {lstm_units}    ||",end="")
                X, y = create_multivariate_dataset(data, min_lag, max_lag)
                mape_values = []

                for train_index, val_index in kf.split(X):
                    X_train, X_val = X[train_index], X[val_index]
                    y_train, y_val = y[train_index], y[val_index]

                    # Building and Training the LSTM Model
                    model = Sequential()
                    model.add(LSTM(lstm_units, activation='relu', input_shape=(X.shape[1], X.shape[2])))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mse')

                    # Training the model
                    model.fit(X_train, y_train, epochs=5, verbose=0)

                    # Making predictions
                    y_pred = model.predict(X_val, verbose=0).flatten()

                    # Calculating MAPE for the fold
                    mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
                    mape_values.append(mape)

                # Averaging MAPE across all folds
                average_mape = np.mean(mape_values)
                print(f" {average_mape}")
                # Update best hyperparameters if current MAPE is lower
                if average_mape < best_mape:
                    best_mape = average_mape
                    best_hyperparams = (min_lag, max_lag, lstm_units)

print(f'Best Hyperparameters: min_lag={best_hyperparams[0]}, max_lag={best_hyperparams[1]}, LSTM Units={best_hyperparams[2]}')
print(f'Best MAPE: {best_mape}%')

# Store the best model's predictions and actual values
best_predictions = []
best_actuals = []

# Retrain the best model
X, y = create_multivariate_dataset(data, *best_hyperparams[:2])
kf = KFold(n_splits=k, shuffle=False)

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Building and Training the LSTM Model with best hyperparameters
    model = Sequential()
    model.add(LSTM(best_hyperparams[2], activation='relu', input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Training the model
    model.fit(X_train, y_train, epochs=5, verbose=0)

    # Making predictions
    y_pred = model.predict(X_val, verbose=0).flatten()

    best_predictions.extend(y_pred)
    best_actuals.extend(y_val)

# Plot the predictions against the actual values
plt.figure(figsize=(10, 6))
plt.plot(best_actuals, label='Actual Sales')
plt.plot(best_predictions, label='Predicted Sales', color='red')
plt.title('Best Model Predictions vs Actual Sales')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.show()
