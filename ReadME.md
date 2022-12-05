# Detection of AF with PPM
In this project, the main goal is to detect AF over Photoplethysmogram (PPG) signals converted in to a matrix following ELEKTRA's pipeline. 
AF is recognized in the electrocardiogram (ECG) as an irregularly irregular rhythm lasting more than 30 s, with no discernible P-waves preceding the QRS complex [[Paulus Kirchof et al, 2016](https://academic.oup.com/eurheartj/article/37/38/2893/2334964?login=true)]. It does not affect equally to the same population. Hence, the most affected population is white men [[Massimo Zoni-Berisso et al, 2014](https://www.dovepress.com/epidemiology-of-atrial-fibrillation-european-perspective-peer-reviewed-fulltext-article-CLEP)].

A photoplethysmogram is a pulse pressure signal resulting from the propagation of blood pressure pulses along arterial blood vessels. Measured on the periphery, it carries rich information about the cardiac activity, cardiovascular condition, the interaction between parasympathetic and sympathetic nervous systems, and hemoglobin level [[A. Resit Kavsaoglu et al, 2015](https://www.sciencedirect.com/science/article/pii/S1568494615002227?via%3Dihub)].

In a PPG signal, AF is manifested as varying pulse-to pulse intervals and pulse morphologies. On the other hand, a normal sinus rhythm (NSR) is recognizable through regularly spaced PPG pulses with similar morphologies between consecutive pulses. Recognizing an arrhythmia in a PPG signal can sometimes be challenging in the presence of artifacts. Differences between a PPG signal with AF or non-AF can be seen here ([source](https://www.nature.com/articles/s41746-019-0207-9)):

![alt text](https://github.com/cfusterbarcelo/PPM/blob/main/images-and-files/ppg_af_nonaf_dif.png)

## Literature
The MIMIC III ICU Data is used in the study performed by [Syed Khairul Bashar et al, 2020](https://ieeexplore.ieee.org/abstract/document/9094371) to evaluate AF over with Electrocardiogram (ECG) signals. A Linear Discriminant Analysis (LDA) is the Machine Learning algorithm used to perfom the classification between AF/nonAF users. In their final results, we can see that accuracies close to 100% (98.99%) are achieved when testing with the database. 

A PPG pulse is modeled by a linear combination of a log-normal and two Gaussian waveforms in [Andrius Solosenko et al, 2017](https://www.sciencedirect.com/science/article/pii/S0010482516303365). Hence, based on RR interval information, they model a new PPG pulse. The results show that the model PPG signals closely resemble real signal for sinus rhythm, premature beats, as well as for AF.

## Datasets
### MIMIC PERform AF Dataset
This dataset can be extracted from [here](https://ppg-beats.readthedocs.io/en/latest/datasets/mimic_perform_af/). It contains 20 minutes of data from 19 patients in AF, and 16 patients in normal sinus rhythm (non-AF). It was used to compare performance between AF and normal sinus rhythm.

In "Detecting beats in the photoplethysmogram: benchmarking open-source algorithms" by [Peter H. Charlton et al, 2022](https://iopscience.iop.org/article/10.1088/1361-6579/ac826d/meta), this database is tested over different PPG detectors. Achieving great results when detecting the PPG beats. 

It seems that there is not more literature regarding this database. 

### Long-term electrocardiogram and wrist-based photoplethysmogram recordings with annotated atrial fibrillation episodes
It is a zenodo dataset download from [here](https://zenodo.org/record/5815074). This dataset contains either ECG and PPG signals in a '.mat' file from 8 patients with AF monitored from 5 to 8 days. The PPG signals are considered to be the ones called 'PPG_GREEN' in this file as a 'green LED embedded' was used to record the PPG signal.

The filtering and extraction of R-peaks it is almost impossible as the signal is too noisy. Even after applying a bandpass filter to the PPG signal, it has been almost impossible to process the file with the [heartpy library](https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/heartpy.heartpy.html#heartpy-main) as it is done for other datasets. 

Resulting images from the analysis of this database can be found as:
* PPG extracted from the mat file as it comes in [here](https://github.com/cfusterbarcelo/PPM/blob/main/images-and-files/mat_ppg.png).
* PPG signal after applying the bandpass filter [here](https://github.com/cfusterbarcelo/PPM/blob/main/images-and-files/mat_ppg_bandpass_filter.png).
* PPG signal with its detected R peaks by the heartpy library [here](https://github.com/cfusterbarcelo/PPM/blob/main/images-and-files/mat_ppg_peaks_extracted.png).

Thus, this database won't be used for the moment.

## Files

* __process-ppg.py__: File used to process PPG signals and convert them into the Photoplethysmatrix (PPM). A database of PPG signals must be processed and a new database of PPM images is extracted and obtained from this file. 
* __utilities.py__: Some functions needed in other python files.
* __move_directory.py__: Once the dataset of PPM images is obtained, depending on how the model will be ran, a different structure of directories is needed. Hence, this file is to perform train/test split to then feed the images into a CNN. 
* __binary_classification.py__: The CNN is constructed here. Depending if it is a one layer or a two layer CNN some things are needed and other pieces of code are commented. The results from all the experiments are stored in [MimicPerformAF_output/TestX](https://github.com/cfusterbarcelo/PPM/tree/main/MimicPerformAF_output) . Hence, each TestX folder will include a description file explaining the differences between each of the launched experiments. 
* __binary_classification.submit__: File needed to run the binary_classification.py file in Artemisa (the server).
* __activation-maximimsation.py__: Activation Maximisation of a 1-layer CNN is performed in this file. A description of what has been used and the results is provided in its [description file](https://github.com/cfusterbarcelo/PPM/blob/main/MimicPerformAF_output/Test05/Test05-description.md).

The [ppm](https://github.com/cfusterbarcelo/PPM/tree/main/images-and-files) folder includes an example of how the PPM resulting images are. These are extracted from the _.dat_ and _.hea_ files included in that same folder which are from one random user from the MIMIC PERform AF Dataset. In fact, a PPM image looks like this:

![alt text](https://github.com/cfusterbarcelo/PPM/blob/main/images-and-files/ppm0.png)
