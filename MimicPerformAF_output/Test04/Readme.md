# Test 04- Description
Results from running a 1-layer CNN for classifying into AF or NON-af users. The split for training and test has been based on splitting **users**. 
Therefore, a binary classification has taken place. 

The database used in this experiments is MimicPerformAF. 

Database is constructed as:
* Train (80%):
    + AF: 15 users from 001 to 015
    + Non_AF: 13 users from 001 to 013
* Test (20%):
    + AF: 4 users from 016 to 019
    + Non_AF: users 014, 015 and 016

## Details
* Number of epochs= 100
* Batchsize= 32
* Target image size = 120,160
* Train split= 80% based on signals
* Validation split= 10%
* Cropping layer= (13, 13),(20,41)
* Number of Conv2D layers = 1

## Code pieces
Some peices of code specific for this file in regards of their nature: a binary classification.
``
    predictions = model.predict(dataset)
    y_pred = np.where(predictions>=0.5, 1, 0)
    y_true = dataset.classes
 ``

``
    model.compile(loss=tf.keras.losses.binary_crossentropy, ...)
``
