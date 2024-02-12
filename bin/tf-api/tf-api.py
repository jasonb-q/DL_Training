    """
    For learning purposes I am to write a short model using TensorFlow's Keras API.
    """
import tensorflow as tf

class LinearRegression(tf.Model):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.built = False

    @tf.function
    def __call__(self, inputs):

if __name__ == "__main__":
