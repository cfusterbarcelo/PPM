# Summary
All launched experiments for this test are launched equally with same parameters. The changing part for each of them are the users that are used to train and test the network.

## Partitions
* **Partition 1**
AF train users: [1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19]
Non AF train users: [1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16]
AF test users: [3, 5, 15, 17]
Non AF test users: [3, 4, 8]


* **Partition 2**
AF train users: [2, 4, 5, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 18, 19]
Non AF train users: [1, 2, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16]
AF test users: [1, 3, 10, 12]
Non AF test users: [3, 5, 8]


* **Partition 3**
AF train users: [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 19]
Non AF train users: [1, 2, 3, 5, 7, 8, 9, 10, 12, 13, 14, 15, 16]
AF test users: [2, 13, 17, 18]
Non AF test users: [4, 6, 11]


* **Partition 4**
AF train users: [2, 3, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19]
Non AF train users: [1, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16]
AF test users: [1, 4, 5, 12]
Non AF test users: [2, 4, 7]


* **Partition 5**
AF train users: [1, 2, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18]
Non AF train users: [2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16]
AF test users: [3, 4, 11, 19]
Non AF test users: [1, 9, 12]


* **Partition 6**
AF train users: [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 17, 18, 19]
Non AF train users: [1, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16]
AF test users: [7, 12, 15, 16]
Non AF test users: [2, 9, 11]


* **Partition 7**
AF train users: [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 17, 18]
Non AF train users: [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 14, 15, 16]
AF test users: [7, 12, 16, 19]
Non AF test users: [7, 12, 13]


* **Partition 8**
AF train users: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 16, 17, 18, 19]
Non AF train users: [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16]
AF test users: [11, 12, 14, 15]
Non AF test users: [1, 3, 13]


* **Partition 9**
AF train users: [1, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18]
Non AF train users: [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16]
AF test users: [2, 6, 17, 19]
Non AF test users: [3, 7, 13]


* **Partition 10**
AF train users: [1, 2, 3, 4, 5, 7, 9, 10, 12, 13, 14, 15, 16, 17, 18]
Non AF train users: [2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]
AF test users: [6, 8, 11, 19]
Non AF test users: [1, 5, 9]




## Model Architecture
```
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

