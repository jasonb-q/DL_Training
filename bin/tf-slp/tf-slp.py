import tensorflow as tf

if __name__ == "__main__":
    """ 
        For learning purposes I am to write a single layer perceptron using TensorFlow.
        Not including any dataloading or anything else, just the model.
    """
    # Print TensorFlow and Keras version
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {tf.keras.__version__}")

    # Define the model
    # Single input layer with 2 inputs and 1 output using sigmoid activation
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(2,), activation="sigmoid")
    ])

    # Compile the model
    model.compile(optimizer="sgd", loss="mean_squared_error")
