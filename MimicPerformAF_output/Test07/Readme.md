# Test 06 - Description
This experiment is considered to be part from the experiments to perform the ablation study where the feasibility of the model and the approach for identification of AF with PPM is evaluated. Hence, the main difference between this test and [Test04](missing-reference) and [Test06](missing-reference) is the number of users that have been used to train and test the network.

The results are obtained from running a 1-layer CNN for classifying into AF or NON-af users. The split for training and test has been based on splitting **users**. 
Therefore, a binary classification has taken place. 

The database used in this experiments is MimicPerformAF. 

Database is constructed as:
* Train (aprox. 60%):
    + AF: 11 users from 001 to 011
    + Non_AF: 9 users from 001 to 009
* Test (aprox. 40%):
    + AF: 8 users from 012 to 019
    + Non_AF: 7 users from 010 to 016

## Details
* This test is repeated two times for two different number of epochs
    *  Number of epochs= 50 and 100
* Batchsize= 32
* Target image size = 120,160
* Train split= 70% based on signals
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
