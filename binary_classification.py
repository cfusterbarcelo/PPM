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
import seaborn as sns
from utilities import obtain_metrics, plotting_metrics, calculating_metrics


# VARS for each CNN launched to change depending on the DDBB
epochs_num = 1
epochs_str = str(epochs_num)+'e'
num_classes = 1
ddbb = 'MimicPerformAF'
batchsize = 32


# To run it from the iMac
current_dir = pathlib.Path(__file__).resolve()
train_path = str(pathlib.Path(current_dir).parents[1] / 'PPM_DDBB/MimicPerformAF/Train/')
test_path = str(pathlib.Path(current_dir).parents[1] / 'PPM_DDBB/MimicPerformAF/Test/')

# To run it from Artemisa
results_path = '/lhome/ext/uc3m0571/PPM/Results/'
pkl_path = '/lhome/ext/uc3m0571/PPM/Results/'

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
strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Cropping2D(cropping=((13, 13),(20,41)), input_shape = input_shape))
    
    # LAYER ONE
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))   
    model.add(tf.keras.layers.Dropout(0.7))

    # LAYER TWO
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
accuracy = np.asarray(train_history.history['accuracy'])
val_accuracy = np.asarray(train_history.history['val_accuracy'])
loss = np.asarray(train_history.history['loss'])
val_loss = np.asarray(train_history.history['val_loss'])
tp = np.asarray(train_history.history['true_positives'])
tn = np.asarray(train_history.history['true_negatives'])
fp = np.asarray(train_history.history['false_positives'])
fn = np.asarray(train_history.history['false_negatives'])
auc = train_history.history['auc']


plotting_metrics(auc)
plotting_metrics(accuracy)
plotting_metrics(val_accuracy)
plotting_metrics(loss)
plotting_metrics(val_loss)

training = 'training'
validation = 'validation'
train_metrics = calculating_metrics(train_path, training, training, model, results_path)
print(train_metrics)
validation_metrics = calculating_metrics(train_path, validation, validation, model, results_path)
print(validation_metrics)
test_metrics = calculating_metrics(test_path, None, 'testing', model, results_path)
print(test_metrics)