import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input,Dense,Reshape,Conv2DTranspose,BatchNormalization,Activation,Flatten
import tensorflow_probability as tfp
import numpy as np

def cnn_model(input_shape):
    inputs = Input(input_shape)
    X = Flatten()(inputs)
    filters = 1024
    print(X.shape)
    X = Dense(8*8*256)(X)
    X = Reshape((8,8,256))(X)

    for i in range(4):
        X = Conv2DTranspose(filters,4,strides=2,padding="same")(X)
        X =BatchNormalization()(X)
        X = Activation('relu')(X)
        filters//=2
    X = Conv2DTranspose(1,4,2,'same')(X)
    X = Activation('sigmoid')(X)
    model = keras.Model(inputs = inputs,outputs = X)
    return model
def train(model,x,y):
    batch_size = tf.shape(y)[0]
    label = np.zeros((batch_size,256,256,1))
    for i in range(batch_size):
        label[i,y[i,0],y[i,1],1]=1
    with tf.GradientTape(persistent = True) as tape:
        pred = model(x)
        loss = tf.reduce_mean(abs(label-pred),axis=(1,2))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
