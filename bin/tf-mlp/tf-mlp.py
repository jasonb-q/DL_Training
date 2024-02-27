import tensorflow as tf
import numpy as np

def custom_mse_loss(y_true, y_pred):
    return tf.math.reduce_mean(tf.math.square(y_true - y_pred))

def custom_dice_loss(y_true, y_pred):
    """Calculate the DICE loss between y_true and y_pred."""
    intersection = np.sum(y_true * y_pred)
    dice = 2*np.sum(intersection)/(np.sum(y_true) + np.sum(y_pred))
    return 1-dice

if __name__ == "__main__":
    """ 
        For learning purposes I am to write a multi layer perceptron using TensorFlow.
        Not including any dataloading or anything else, just the model.
    """
    # Print TensorFlow and Keras version
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {tf.keras.__version__}")

    # Define the model
    # Single input layer with 2 inputs and 1 output using sigmoid activation
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    # Compile the model
    model.compile(optimizer="sgd", loss=custom_dice_loss)

    # Print the model summary
    model.summary()