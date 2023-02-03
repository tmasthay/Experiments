import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def model1(num_hidden_layers=20, optimizer='sgd', loss='mean_squared_error'):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
    #model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    for l in range(num_hidden_layers):
        model.add(layers.Conv2D(64, (3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

    model.add(layers.Conv2D(3, (3,3), padding='same'))

    model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])

    return model