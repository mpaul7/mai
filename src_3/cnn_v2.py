import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Simulated Data (Replace this with your actual DataFrame)
# Assuming your DataFrame has 57 statistical features and a target column named 'label'
num_samples = 1000
num_features = 5
data = pd.DataFrame(
    np.random.rand(num_samples, num_features),
    columns=[f'feature_{i}' for i in range(num_features)]
)
data['label'] = np.random.choice(['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6'], num_samples)

# Step 1: Preprocess the data
# Separate features and labels
X = data.iloc[:, :-1].values  # Feature columns
print(X.shape)
print(X)
y = data['label'].values      # Target column

# Encode labels to integers and then to one-hot
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape input for CNN (CNN expects 3D input: samples, timesteps, features)
X_reshaped = X_scaled[:, :, np.newaxis]  # Adding a new axis for timesteps (1 in this case)
print(X_reshaped.shape)
print(X_reshaped)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_onehot, test_size=0.2, random_state=42)

# Step 2: Build the CNN Model
# input_layer = Input(shape=(num_features, 1))  # 57 features, 1 "timestep"
input_layer = {name: Input(shape=(num_features, 1), dtype=tf.float32, name=name)for name in ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4']}
print(input_layer)

# Convolutional Layer
x = Conv1D(filters=32, kernel_size=3, activation='relu')(input_layer)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

# Add another convolutional layer
x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

# Flatten and Fully Connected Layers
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)

# Output Layer
output_layer = Dense(7, activation='softmax')(x)  # 7 output units for classification

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)
tf.keras.utils.plot_model(model, "/home/mpaul/projects/mpaul/mai/src_3/cnn_v2.png", show_shapes=True)
model.summary()

# Step 3: Compile the Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train the Model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# Step 5: Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Step 6: Make Predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
print(f"Predicted classes: {predicted_classes}")
