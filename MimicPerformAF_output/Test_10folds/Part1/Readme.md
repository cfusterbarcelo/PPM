# Experiment - Partition 1

## Description of the Experiment
This experiment was conducted in response to a reviewer's suggestion in our paper, addressing concerns about the partitioning of train and test data potentially affecting the results. To address this, we performed 10 different partitions with randomized users and patients.

## User Partitioning
- **AF train users:** [1, 2, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 19]
- **Non AF train users:** [1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16]
- **AF test users:** [3, 5, 15, 17]
- **Non AF test users:** [3, 4, 8]

## Training and Validation Info
Found 1512 images belonging to 2 classes.
Found 168 images belonging to 2 classes.
Found 420 images belonging to 2 classes.
Batch shape=(32, 120, 160, 3), min=0.000, max=1.000
Epochs 100

## Model Architecture
```python	
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
```

## Evaluation Results

| Metric     | Class 0 | Class 1 | Total |
|------------|---------|---------|-------|
| True Pos.  | 240.0   | 180.0   | 420.0 |
| True Neg.  | 180.0   | 240.0   | 420.0 |
| False Pos. | 0.0     | 0.0     | 0.0   |
| False Neg. | 0.0     | 0.0     | 0.0   |
| FAR        | 0.0     | 0.0     | 0.0   |
| FRR        | 0.0     | 0.0     | 0.0   |
| Accuracy   | 1.0     | 1.0     | 1.0   |

