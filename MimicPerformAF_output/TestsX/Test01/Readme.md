# Description
Results from running a 2-layer CNN for classifying into AF or NON-af users. The split for training and test has been based on splitting signals instead of users. 
Therefore, a binary classification has taken place. 

The database used in this experiments is MimicPerformAF.
## Details
* Number of epochs= 100
* Batchsize= 32
* Target image size = 120,160
* Train split= 80% based on signals
* Validation split= 10%
* Cropping layer= (13, 13),(20,41)
* Number of Conv2D layers = 2

## Code pieces
Some peices of code specific for this file in regards of their nature: a binary classification.
``
    predictions = model.predict(dataset)
    y_pred = np.where(predictions>=0.5, 1, 0)
    y_true = dataset.classes
 ``

``
    # LAYER TWO
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.7))
``

``
    model.compile(loss=tf.keras.losses.binary_crossentropy, ...)
``
