# Define the model
my_model = XGBRegressor(n_estimatiors = 1000, learning_rate = 0.05) # Your code here

# Fit the model
my_model.fit(X_train, y_train) # Your code here

# Get predictions
predictions = my_model.predict(X_valid) # Your code here

# Calculate MAE
mae = mean_absolute_error(predictions, y_valid) # Your code here

print("Mean Absolute Error:" , mae)
