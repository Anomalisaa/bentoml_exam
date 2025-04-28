import os
import pandas as pd
import bentoml
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Define paths
data_dir = os.path.join("data", "processed")

# Load processed data
X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).squeeze()
y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).squeeze()

# Build DNN model
input_dim = X_train.shape[1]
model = Sequential([
    Input(shape=(input_dim,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
# Compile the model
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mse']
)

# Train the model
model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=32,
    verbose=2
)

# Evaluate performance on test set
preds = model.predict(X_test).squeeze()
r2 = r2_score(y_test, preds)
rmse = mean_squared_error(y_test, preds, squared=False)
print("Model performance on test set:")
print(f"- R2 score: {r2:.4f}")
print(f"- RMSE: {rmse:.4f}")

# Save the trained model to BentoML Model Store
model_ref = bentoml.tensorflow.save_model("admissions_dense_nn", model)
print(f"Model saved to BentoML with reference: {model_ref}")
