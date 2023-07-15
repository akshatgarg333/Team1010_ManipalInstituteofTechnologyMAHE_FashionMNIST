#Creating the model
def create_model():
    model = tf.keras.Sequential([                        # Model is having three sequential layers
    tf.keras.layers.Flatten(input_shape=(28, 28)),   # First layer converts the two dimensional 28X28 matrix into a one dimensional matrix of 784 pixels
    tf.keras.layers.Dense(128, activation='relu'),   # First neural layer is having 128 neurons
    tf.keras.layers.Dense(10, activation='softmax')  # Each of the 10 nodes tell which label the data belongs to
])


    return model
