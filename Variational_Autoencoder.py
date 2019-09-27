#Loading the libraries and the dataset
from keras.datasets import fashion_mnist

%pylab inline
import os
import keras
import numpy as np
import pandas as pd
import keras.backend as K

from time import time
from sklearn.cluster import KMeans
from keras import callbacks
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Input, Lambda, Layer, Add, Multiply
from keras.initializers import VarianceScaling
from keras.engine.topology import Layer, InputSpec

from scipy.misc import imread
from sklearn.metrics import accuracy_score, normalized_mutual_info_score

(train_x, train_y), (val_x, val_y) = fashion_mnist.load_data()

train_x = train_x/255.
val_x = val_x/255.

train_x = train_x.reshape(-1, 784)
val_x = val_x.reshape(-1, 784)

# this is our input placeholder
input_img = Input(shape=(784,))

# "encoded" is the encoded representation of the input
encoded = Dense(500, activation='relu')(input_img)

z_mu = Dense(10)(encoded)
z_log_sigma = Dense(10)(encoded)

class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_sigma = inputs

        kl_batch = - .5 * K.sum(1 + log_sigma -
                                K.square(mu) -
                                K.exp(log_sigma), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs

z_mu, z_log_sigma = KLDivergenceLayer()([z_mu, z_log_sigma])
z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_sigma)

eps = Input(tensor=K.random_normal(shape=(K.shape(input_img)[0],10)))
z_eps = Multiply()([z_sigma, eps])
z = Add()([z_mu, z_eps])

decoder = Sequential([
    Dense(500, input_dim=10, activation='relu'),
    Dense(784, activation='sigmoid')
])

decoded = decoder(z)

# this model maps an input to its reconstruction
autoencoder = Model([input_img, eps], decoded)

autoencoder.summary()

def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

autoencoder.compile(optimizer='adam', loss=nll)

estop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

train_history = autoencoder.fit(train_x, train_x, epochs=500, batch_size=2048, validation_data=(val_x, val_x), callbacks=[estop])

pred = autoencoder.predict(val_x)

plt.imshow(pred[0].reshape(28, 28), cmap='gray')

plt.imshow(val_x[0].reshape(28, 28), cmap='gray')