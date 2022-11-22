# Description - Test 05

In this test activation maximisation is studdied. 

## Activation Maximimsation
As it is explained in [this article](https://towardsdatascience.com/every-ml-engineer-needs-to-know-neural-network-interpretability-afea2ac0824e):

> Activation Maximization is a method to visualize neural networks, and aims to maximize the activation of certain neurons. During normal training, one would iteratively tune the weights and biases of the network such that the error, or loss, of the neural network is minimized across training examples in the data set. On the other hand, activation maximization flips this around: after the classifier has been trained, we want to iteratively find the parts of the data that the model thinks belongs to a class.
>
> -- <cite>TowardsDataScience</cite>

What we are doing is removing last layer from the CNN (in this case, the a Sigmoid function) and substitute it for a new layer. The Loss function will be maximised to overadjust our model to a noise image (created on purpose) to show where the activation of the model takes place, where does the CNN focus to perform the classification between AF and non AF.

## Model Loaded
The loaded model is the one from __Test04__ which is a one-layer CNN with all details explained in the description file. 
The database used to train and test this model is the Mimic Perform AF separated into users for the training/test subsets.

## Results

Resulting image where the activation can be seen is the one following:

![alt text](https://github.com/cfusterbarcelo/PPM/blob/main/MimicPerformAF_output/Test05/act-max.png)
