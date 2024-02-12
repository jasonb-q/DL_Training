import tensorflow as tf

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
    model.compile(optimizer="sgd", loss="mean_squared_error")

    # Print the model summary
    model.summary()