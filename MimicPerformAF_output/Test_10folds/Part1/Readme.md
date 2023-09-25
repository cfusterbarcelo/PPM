# Experiment Partition 1 README

## Description of the Experiment
This experiment was conducted in response to a reviewer's suggestion in our paper, addressing concerns about the partitioning of train and test data potentially affecting the results. To address this, we performed 10 different partitions with randomized users and patients.

## User Partitioning
- **AF Train Users (MimicPerformAF_Part1):** [19, 10, 2, 18, 14, 4, 6, 12, 8, 5, 11, 3, 7, 1, 13]
- **Non_AF Train Users (MimicPerformAF_Part1):** [5, 14, 13, 9, 3, 6, 7, 11, 8, 1, 10, 15, 16]
- **AF Test Users (MimicPerformAF_Part1):** [16, 15, 9, 17]
- **Non_AF Test Users (MimicPerformAF_Part1):** [12, 4, 2]

## Training and Validation Info
- Found 1512 images belonging to 2 classes.
- Found 168 images belonging to 2 classes.
- Found 420 images belonging to 2 classes.
- Batch shape=(32, 120, 160, 3), min=0.000, max=1.000
- Epoch 1/100

## Model Architecture
```plaintext
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
cropping2d (Cropping2D)     (None, 94, 99, 3)         0         
                                                                 
conv2d (Conv2D)             (None, 92, 97, 64)        1792      
                                                                 
max_pooling2d (MaxPooling2D  (None, 46, 48, 64)       0         
)                                                                
                                                                 
dropout (Dropout)           (None, 46, 48, 64)        0         
                                                                 
flatten (Flatten)           (None, 141312)            0         
                                                                 
dense (Dense)               (None, 256)               36176128  
                                                                 
dense_1 (Dense)             (None, 1)                 257       
                                                                 
=================================================================
Total params: 36,178,177
Trainable params: 36,178,177
Non-trainable params: 0
_________________________________________________________________
Found 1512 images belonging to 2 classes.
``` 

## Evaluation Results

| Metric                     | Value |
|----------------------------|-------|
| Accuracy                   | 1.0   |
| False Positives (FP)       | [0.0, 0.0] |
| False Positives Sum        | 0.0   |
| False Negatives (FN)       | [0.0, 0.0] |
| False Negatives Sum        | 0.0   |
| True Positives (TP)        | [240.0, 180.0] |
| True Positives Sum         | 420.0 |
| True Negatives (TN)        | [180.0, 240.0] |
| True Negatives Sum         | 420.0 |
| False Acceptance Rate (FAR)| [0.0, 0.0] |
| FAR Mean                   | 0.0   |
| False Rejection Rate (FRR) | [0.0, 0.0] |
| FRR Mean                   | 0.0   |
