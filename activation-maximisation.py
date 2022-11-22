''''''
'''_______________________________________________
    Python file to perform activation-maximisation
__________________________________________________
## Author: Caterina Fuster Barceló
## Version: 1.0
## Email: cafuster@pa.uc3m.es
## Status: Development
__________________________________________________

__________________________________________________
            For this version 1.0
__________________________________________________
## Database used: MIMIC PERform AF Dataset
## Input files: model
## Output files: .png
__________________________________________________'''
# pip install tf-keras-vis

import tensorflow as tf
from tf_keras_vis.activation_maximization import ActivationMaximization
import matplotlib.pyplot as plt
import numpy as np
import pathlib

results_path = '/Users/caterina/Library/CloudStorage/GoogleDrive-cafuster@pa.uc3m.es/La meva unitat/COSEC/PPG/PPM/MimicPerformAF_output/Test05/'
model_files =  '/Users/caterina/Library/CloudStorage/GoogleDrive-cafuster@pa.uc3m.es/La meva unitat/COSEC/PPG/PPM/MimicPerformAF_output/Test05/model'

# Load a trained model
model = tf.keras.models.load_model(model_files)

def loss(output):
  return (output[0, 0])

def model_modifier(m):
    m.layers[-1].activation = tf.keras.activations.linear

# Initialize Activation Maximization
visualize_activation = ActivationMaximization(model, model_modifier)

# Generate a random seed for each activation - noise
seed_input = tf.random.uniform((1, 120, 160, 3), 0, 1)

# Generate activations and convert into images
activations = visualize_activation(loss, seed_input=seed_input, steps=512)
images = [activation.astype(np.float32) for activation in activations]

# Visualize each image
for i in range(0, len(images)):
  visualization = images[i].reshape(120,160, 3)
  plt.imshow(visualization)
  plt.title(f'Target = {i}')
  plt.savefig(results_path +'/act-max.png')
  plt.show()
