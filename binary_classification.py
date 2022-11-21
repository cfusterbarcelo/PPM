#!/lhome/ext/uc3m057/uc3m0571/miniconda3/envs/ELEKTRA/bin/python
# -*- coding: utf-8 -*-
''''''
'''_______________________________________________
    Python file to perform BINARY classification
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
## Input files: .png
## Output files: .pkl, .out, .txt
__________________________________________________'''

import pathlib
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# from utilities import plotting_metrics, calculating_metrics
import sys
import os
import pickle 
from sklearn.metrics import confusion_matrix

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# VARS for each CNN launched to change depending on the DDBB
epochs_num = 50
epochs_str = str(epochs_num)+'e'
num_classes = 1
ddbb = 'MimicPerformAF'
batchsize = 32
test = 'Test05'

# To run it from the iMac
# current_dir = pathlib.Path(__file__).resolve()
# train_path = str(pathlib.Path(current_dir).parents[1] / 'PPM_DDBB/MimicPerformAF/Train/')
# test_path = str(pathlib.Path(current_dir).parents[1] / 'PPM_DDBB/MimicPerformAF/Test/')

# To run it from Artemisa
train_path = '/lhome/ext/uc3m057/uc3m0571/PPM/DDBB/MimicPerformAF_bu/Train/'
test_path = '/lhome/ext/uc3m057/uc3m0571/PPM/DDBB/MimicPerformAF_bu/Test/'
results_path = '/lhome/ext/uc3m057/uc3m0571/PPM/Results/MimicPerformAF/' + test + '/'
output_path = '/lhome/ext/uc3m057/uc3m0571/PPM/PPM/MimicPerformAF_output/' + test + '/'
output_file = output_path + test + '-outcome.txt'

orig_stdout = sys.stdout
f = open(output_file, 'w')
sys.stdout = f

# ==== DATA INICIALIZATION ====

datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.1, rescale=1./255)

train_dataset = datagen.flow_from_directory(train_path, 
                                       target_size=(120,160),
                                       class_mode='binary',
                                       seed=0,
                                       subset='training')

validation_dataset = datagen.flow_from_directory(train_path, 
                                       target_size=(120,160),
                                       class_mode='binary',
                                       seed=0,
                                       subset='validation')

test_dataset = datagen.flow_from_directory(test_path, 
                                       target_size=(120,160),
                                       class_mode='binary',
                                       seed=0)

batchX, batchy = train_dataset.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))


input_shape = [batchX.shape[1], batchX.shape[2], batchX.shape[3]]

# Visualisation of one cropped picture
# one_cropped_pic = batchX[1,13:120-13,20:160-41]
# plt.imshow(one_cropped_pic)


# ====== TRAINING  ======
# strategy = tf.distribute.MultiWorkerMirroredStrategy()
# with strategy.scope():
model = tf.keras.Sequential()
model.add(tf.keras.layers.Cropping2D(cropping=((13, 13),(20,41)), input_shape = input_shape))

# LAYER ONE
# model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))   
# model.add(tf.keras.layers.Dropout(0.7))

# LAYER TWO
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.7))

# LAYER THREE
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='sigmoid'))

model.compile(loss=tf.keras.losses.binary_crossentropy, 
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy',
                    tf.keras.metrics.TruePositives(),
                    tf.keras.metrics.TrueNegatives() , 
                    tf.keras.metrics.FalsePositives(), 
                    tf.keras.metrics.FalseNegatives(),
                    tf.keras.metrics.AUC()])

# Save after each epoch
SAEE_file = results_path + 'saee'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath= SAEE_file,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_freq = 'epoch',
    save_best_only=True)

log_dir = results_path + 'log'
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, 
    histogram_freq=1)

train_history = model.fit(
    x = train_dataset,
    batch_size=batchsize,
    epochs=epochs_num,
    steps_per_epoch=len(train_dataset)//batchsize,
    verbose=1,
    validation_data=validation_dataset,
    callbacks=[model_checkpoint_callback, tensorboard_callback]
)

# ====== SAVE AND TEST THE MODEL  ======
model_file = results_path + 'model' + ddbb + epochs_str
model.save(model_file)

scores = model.evaluate(test_dataset, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.summary()


# ===== METRICS CALCULATION ===========
def obtain_metrics(conf_m):
    fp = conf_m.sum(axis=0) - np.diag(conf_m)
    fn = conf_m.sum(axis=1) - np.diag(conf_m)
    tp = np.diag(conf_m)
    tn = conf_m.sum() - (fp + fn + tp)
    fp = fp.astype(float)
    fn = fn.astype(float)
    tp = tp.astype(float)
    tn = tn.astype(float)
    fp_n = np.nan_to_num(fp)
    fn_n = np.nan_to_num(fn)
    tp_n = np.nan_to_num(tp)
    tn_n = np.nan_to_num(tn)
    return fp_n, fn_n, tp_n, tn_n

def plotting_metrics(to_plot, results_path, metric):
    plt.figure()
    plt.plot(to_plot)
    plt.title(metric)
    plt.xlabel('epochs')
    plt.savefig(results_path + metric + '.png')
    plt.close()

def calculating_metrics(dataset_path, dataset_subset, dataset_name, model, results_path):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.1, rescale=1. / 255)
    dataset = datagen.flow_from_directory(dataset_path,
                                            target_size=(120,160),
                                            class_mode='binary',
                                            shuffle=False,
                                            seed=0,
                                            subset=dataset_subset)
    predictions = model.predict(dataset)
    y_pred = np.where(predictions>=0.5, 1, 0)
    y_true = dataset.classes
    conf_m = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 20))
    df_cm = pd.DataFrame(conf_m)
    sns.set(font_scale=1)  # for label size
    sns.heatmap(df_cm, annot=True, fmt='g')  # font size
    plt.savefig(results_path + dataset_name +'-confusion-matrix.png')
    plt.close()
    #Obtaining TP, FP, TN and FN and store it in a dictionary
    fp, fn, tp, tn = obtain_metrics(conf_m)
    far = fp/ np.nan_to_num((fp + tn)) * 100
    frr = fn / np.nan_to_num((fn + tp)) * 100
    metrics = {
        'fp': fp,
        'fp_sum': sum(fp), 
        'fn': fn,
        'fn_sum': sum(fn),
        'tp': tp,
        'tp_sum': sum(tp),
        'tn': tn,
        'tn_sum': sum(tn),
        'far': far,
        'far_mean': np.mean(far),
        'frr': frr,
        'frr_mean': np.mean(frr)
    }
    pkl_file = output_path + dataset_name + '.pkl'
    with open(pkl_file, 'wb') as handle:
        pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return metrics


accuracy = np.asarray(train_history.history['accuracy'])
val_accuracy = np.asarray(train_history.history['val_accuracy'])
loss = np.asarray(train_history.history['loss'])
val_loss = np.asarray(train_history.history['val_loss'])
tp = np.asarray(train_history.history['true_positives'])
tn = np.asarray(train_history.history['true_negatives'])
fp = np.asarray(train_history.history['false_positives'])
fn = np.asarray(train_history.history['false_negatives'])
auc = train_history.history['auc']


plotting_metrics(auc, results_path, 'auc')
plotting_metrics(accuracy, results_path, 'accuracy')
plotting_metrics(val_accuracy, results_path, 'validation accuracy')
plotting_metrics(loss, results_path, 'loss')
plotting_metrics(val_loss, results_path, 'validation loss')

training = 'training'
validation = 'validation'
train_metrics = calculating_metrics(train_path, training, training, model, results_path)
print(train_metrics)
print('===================================================================================')

validation_metrics = calculating_metrics(train_path, validation, validation, model, results_path)
print(validation_metrics)
print('===================================================================================')

test_metrics = calculating_metrics(test_path, None, 'testing', model, results_path)
print(test_metrics)
print('===================================================================================')

########################### EXPLAINABLE AI

# pip install tf-keras-vis

from tf_keras_vis.activation_maximization import ActivationMaximization
import matplotlib.pyplot as plt

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
  visualization = images[i].reshape(120,160)
  plt.imshow(visualization)
  plt.title(f'Target = {i}')
  plt.savefig(results_path + dataset_name +'-act-max.png')
  plt.show()
