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

# VARS for each CNN launched to change depending on the DDBB
epochs_num = 1
epochs_str = str(epochs_num)+'e'
num_classes = 1
bpf = 5
ddbb = 'MimicPerformAF'
batchsize = 32

# To run it from the iMac
current_dir = pathlib.Path(__file__).resolve()
train_path = str(pathlib.Path(current_dir).parents[1] / 'PPM_DDBB/MimicPerformAF/Train/')
test_path = str(pathlib.Path(current_dir).parents[1] / 'PPM_DDBB/MimicPerformAF/Test/')

# To run it from Artemisa


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

train_history = model.fit(
    x = train_dataset,
    batch_size=batchsize,
    epochs=epochs_num,
    steps_per_epoch=len(train_dataset)//batchsize,
    verbose=1,
    validation_data=validation_dataset
)

