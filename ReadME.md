# Detection of AF with PPM
In this project, the main goal is to detect AF over Photoplethysmogram (PPG) signals converted in to a matrix following ELEKTRA's pipeline. 

## Literature
The MIMIC III ICU Data is used in the study performed by [Syed Khairul Bashar et al, 2020](https://ieeexplore.ieee.org/abstract/document/9094371) to evaluate AF over with Electrocardiogram (ECG) signals. A Linear Discriminant Analysis (LDA) is the Machine Learning algorithm used to perfom the classification between AF/nonAF users. In their final results, we can see that accuracies close to 100% (98.99%) are achieved when testing with the database. 

## Datasets
### MIMIC PERform AF Dataset
This dataset can be extracted from [here](https://ppg-beats.readthedocs.io/en/latest/datasets/mimic_perform_af/). It contains 20 minutes of data from 19 patients in AF, and 16 patients in normal sinus rhythm (non-AF). It was used to compare performance between AF and normal sinus rhythm.

In "Detecting beats in the photoplethysmogram: benchmarking open-source algorithms" by [Peter H. Charlton et al, 2022](https://iopscience.iop.org/article/10.1088/1361-6579/ac826d/meta), this database is tested over different PPG detectors. Achieving great results when detecting the PPG beats. 

It seems that there is not more literature regarding this database. 

## Files

* __process-ppg.py__: File used to process PPG signals and convert them into the Photoplethysmatrix (PPM). A database of PPG signals must be processed and a new database of PPM images is extracted and obtained from this file. 
* __utilities.py__: Some functions needed in other python files.
* __move_directory.py__: Once the dataset of PPM images is obtained, depending on how the model will be ran, a different structure of directories is needed. Hence, this file is to perform train/test split to then feed the images into a CNN. 
* __binary_classification.py__: The CNN is constructed here. Depending if it is a one layer or a two layer CNN some things are needed and other pieces of code are commented. The results from all the experiments are stored in [MimicPerformAF_output/TestX](https://github.com/cfusterbarcelo/PPM/tree/main/MimicPerformAF_output) . Hence, each TestX folder will include a description file explaining the differences between each of the launched experiments. 
* __binary_classification.submit__: File needed to run the binary_classification.py file in Artemisa (the server).
* __activation-maximimsation.py__: Activation Maximisation of a 1-layer CNN is performed in this file. A description of what has been used and the results is provided in its [description file](https://github.com/cfusterbarcelo/PPM/blob/main/MimicPerformAF_output/Test05/Test05-description.md).

The [ppm](https://github.com/cfusterbarcelo/PPM/tree/main/ppms) folder includes an example of how the PPM resulting images are. These are extracted from the _.dat_ and _.hea_ files included in that same folder which are from one random user from the MIMIC PERform AF Dataset. In fact, a PPM image looks like this:

![alt text](https://github.com/cfusterbarcelo/PPM/blob/main/ppms/ppm0.png)