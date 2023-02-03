import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import model_from_json
from tensorflow.keras.initializers import glorot_uniform
import matplotlib.pyplot as plt
from add_noise import add_noise
import create_model
import os
import numpy as np

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

noise_level = 0.0
noisy_train_images = add_noise(train_images, 0.1)
noisy_test_images = add_noise(test_images, 0.1)

noise_train = noisy_train_images - train_images
noise_test = noisy_test_images - test_images

print(noisy_train_images.shape)
print(noise_train.shape)

plt.figure(figsize=(10,10))
num_plots = 5
ordered_data = [train_images, noisy_train_images, noise_train]
num_subplots = len(ordered_data)
for i in range(num_plots):
    for j in range(num_subplots):
        plt.subplot(num_plots,num_subplots,num_subplots*i+j+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(ordered_data[j][i])
plt.savefig('sanity_plot_noise.pdf')

optimizer = 'adam'
loss = 'mean_squared_error'
if( os.path.exists('json_models/model1.json') ):
    with open('json_models/model1.json', 'r') as json_file:
        print('Loading model1!')
        model = model_from_json(json_file.read())
        model.load_weights('json_models/model1_weights.h5')
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        print('JSON model loaded and compiled successfully!')
        num_examples = 10
        my_subindices = np.random.choice(range(noisy_test_images.shape[0]), num_examples)
        my_subset = noisy_test_images[my_subindices, :, :, :]
        predicted_noise = model.predict(my_subset)
        for (i,j) in enumerate(my_subindices):
            plt.subplot(1,3,1)
            plt.imshow(test_images[j])
            plt.title('Ground Truth')
            plt.subplot(1,3,2)
            plt.imshow(my_subset[i])
            plt.title('Noisy input')
            plt.subplot(1,3,3)
            plt.imshow(my_subset[i] - predicted_noise[i])
            plt.title('Predicted image')
            plt.savefig('predictions/prediction%d.pdf'%i)
            plt.clf()
else:
    model = create_model.model1(5, optimizer, loss)
    print('Model created from scratch!')

model.summary()

history = model.fit(noisy_train_images, noise_train, 
    epochs=1, 
    validation_data=(noisy_test_images, noise_test)
)
print(history)

json_model = model.to_json()
with open('json_models/model1.json', 'w') as json_file:
    json_file.write(json_model)
model.save_weights('json_models/model1_weights.h5')




