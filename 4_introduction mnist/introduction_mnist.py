# -*- coding: utf-8 -*-
"""
Created on Mon May  8 09:10:59 2017

@author: darren
"""

from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
data = mnist.train.next_batch(1)


data_picturn = data[0]
reshape_data = (data_picturn.reshape([28,28])*256).astype(np.int8)
array_to_image = Image.fromarray(reshape_data,'L')
array_to_image.save("test.png")
data_label = data[1]

train_numbers = mnist.train.num_examples
test_numbers = mnist.test.num_examples
validation_numbers = mnist.validation.num_examples

test_all_images = mnist.test.images[:1024]
test_all_labels = mnist.test.labels

