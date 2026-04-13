import os
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from data_loader import load_data
from model import build_model
from generator import batch_generator


def train():
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    print("Building model...")
    model = build_model()

    batch_size = 32
    epochs = 30
    samples_per_epoch = len(X_train)

    print(f"\nTraining configuration:")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_test)}")

    # Shuffle once before generator-based training.
    X_train, y_train = shuffle(X_train, y_train)

    train_generator = batch_generator(X_train, y_train, batch_size, is_training=True)
    validation_generator = batch_generator(X_test, y_test, batch_size, is_training=False)

    # Stop early when validation loss plateaus.
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        min_delta=0.0001,
        restore_best_weights=True
    )

    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.h5')

    # Save best checkpoint during training.
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_loss',
        save_best_only=True
    )

    print("\nStarting training...\n")

    history = model.fit(
        train_generator,
        # Multiplied because training generator uses augmentation.
        steps_per_epoch=(samples_per_epoch // batch_size) * 2,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(X_test) // batch_size,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    model.save(model_path)
    print(f"Model saved as {model_path}")

    plot_training_history(history)

    return model


def plot_training_history(history):
    # Compare train vs validation loss across epochs.
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.title("Loss")
    plt.savefig("loss.png")
    plt.show()


if __name__ == "__main__":
    train()
