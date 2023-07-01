import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

f_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels),(test_images, test_labels)= f_mnist.load_data()

