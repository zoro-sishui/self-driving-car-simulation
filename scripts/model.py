import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import Adam


def nvidia_model():
    """
    NVIDIA autonomous driving model architecture
    Input: (66, 200, 3)
    Output: Steering angle
    """

    model = Sequential([
        # Normalization layer
        # Lambda(lambda x: x / 255.0, input_shape=(66, 200, 3)),
        # Convolutional layers
        Conv2D(24, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),

        # Fully connected layers
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.5),  # helps prevent overfitting
        Dense(50, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1)  # output steering angle
    ])

    return model


def build_model():
    """
    Build and compile the model
    """
    model = nvidia_model()

    model.compile(
        optimizer=Adam(learning_rate=1e-4), 
        loss='mse',
        metrics=['mae']
    )

    model.summary()
    return model