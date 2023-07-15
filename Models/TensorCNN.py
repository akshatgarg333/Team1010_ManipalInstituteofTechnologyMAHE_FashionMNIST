#Function for creating the model
def create_model():
    model = keras.Sequential([
        #Total 20 layers are there in this CNN model having multiple cn layer, dropout layers, dense layers and batch normalisation layers
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)), #Convolutional layer with 32 filters and a relu activation function
        keras.layers.BatchNormalization(), #Layer to normalise the batch
        keras.layers.Dropout(0.3), #Layer to stop 30% of the neurons randomly

        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'), #Convolutional layer with 64 filters and a relu activation function
        keras.layers.MaxPooling2D(pool_size=(2, 2)),#Layer to normalise the batch
        keras.layers.Dropout(0.3),#Layer to stop 30% of the neurons randomly

        keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'), #Convolutional layer with 128 filters and a relu activation function
        keras.layers.BatchNormalization(),#Layer to normalise the batch
        keras.layers.Dropout(0.3),#Layer to stop 30% of the neurons randomly

        keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),  #Convolutional layer with 256 filters and a relu activation function
        keras.layers.BatchNormalization(),#Layer to normalise the batch
        keras.layers.Dropout(0.3),#Layer to stop 30% of the neurons randomly

        keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu'),  #Convolutional layer with 512 filters and a relu activation function
        keras.layers.BatchNormalization(),#Layer to normalise the batch
        keras.layers.Dropout(0.3),#Layer to stop 30% of the neurons randomly

        keras.layers.Flatten(), #Layer to flatten the 2d feature to 1d vector
        keras.layers.Dense(512, activation='relu'), #A fully connected layer with 512 neurons and a relu function
        keras.layers.BatchNormalization(),#Layer to normalise the batch
        keras.layers.Dropout(0.5),#Layer to stop 50% of the neurons randomly
        keras.layers.Dense(128, activation='relu'), #A fully connected layer with 128 neurons and a relu function
        keras.layers.BatchNormalization(),#Layer to normalise the batch
        keras.layers.Dropout(0.5),#Layer to stop 50% of the neurons randomly
        keras.layers.Dense(64, activation='relu'), #A fully connected layer with 64 neurons and a relu function
        keras.layers.BatchNormalization(),#Layer to normalise the batch
        keras.layers.Dropout(0.5),#Layer to stop 50% of the neurons randomly
        keras.layers.Dense(10, activation='softmax') #A fully connected layer with 10 neurons and a relu function to give the output labels
    ])

    return model

