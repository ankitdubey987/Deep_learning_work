# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 10:40:37 2020

@author: Ankit
Deep learning CNN architecture
"""

import tensorflow as tf

# defining the input layer inputs
inputs = tf.keras.layers.Input(shape=(28,28,3))

# model architecture is as follows

c1 = tf.keras.layers.Conv2D(32,kernel_size=5,activation='relu',strides=(1,1),padding='same')(inputs)
p1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(c1)
c2 = tf.keras.layers.Conv2D(64,kernel_size=5,activation='relu', strides=(1,1))(p1)
p2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(c2)

outputs = tf.keras.layers.Dense(units=1024, activation='softmax')(p2)


model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', metrics=['accuracy'],loss='binary_crossentropy')
model.summary()