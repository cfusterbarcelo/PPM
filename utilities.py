#!/lhome/ext/uc3m057/uc3m0571/miniconda3/envs/ELEKTRA/bin/python
# -*- coding: utf-8 -*-
""""""
"""_______________________________________________
   Utilities file
__________________________________________________
## Author: Caterina Fuster Barceló
## Version: 1.0
## Email: cafuster@pa.uc3m.es
## Status: Development
__________________________________________________

__________________________________________________
            For this version 1.0
__________________________________________________"""

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
from torchvision import transforms


# == PROCESS PPG AND OBTAIN PPM ==
def normalise(signal):
    a, b = -1, 1
    c = b - a
    aux = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    norm_signal = c * aux + a
    return norm_signal


def mean_peak_dist(peaks):
    dist = []
    for i in range(len(peaks)):
        if peaks[i] == peaks[-1]:
            break
        d = peaks[i + 1] - peaks[i]
        if i == 0:
            dist.append(d)
            continue
        if d > np.mean(dist) + np.std(dist) * 2:
            continue
        else:
            dist.append(d)
    return np.mean(dist)


def building_matrix(mean_dist, peaks, norm_ppg, init_window, window_samp):
    all_segments = []
    init_seg = int(0.2 * mean_dist)
    fin_seg = int(1.3 * mean_dist)
    peaks = np.asarray(peaks)
    idx = np.where((peaks > init_window) & (peaks < init_window + window_samp))[0]
    # We need to grab a certain number of peaks depending on the window
    # How many peaks are there in the selected window?
    for peak in peaks[idx]:
        if peak - init_seg < 0:
            segment = norm_ppg[0 : peak + fin_seg]
        else:
            segment = norm_ppg[peak - init_seg : peak + fin_seg]
        all_segments.append(segment[:, np.newaxis])
    if all_segments[0].shape[0] < all_segments[1].shape[0]:
        zeros = np.zeros(int(all_segments[1].shape[0]) - int(all_segments[0].shape[0]))[
            :, np.newaxis
        ]
        new_segment = np.concatenate((zeros, all_segments[0]))
        all_segments[0] = new_segment
    # Removing last segment to compute matrix in case sizes do not match
    if len(all_segments[-1]) < len(all_segments[-2]):
        all_segments.pop()
    try:
        matrix = np.concatenate(all_segments, 1)
    except ValueError:
        print("MATRIX CANNOT BE COMPUTED!!!!")
        return None
    return matrix.T


# == CNN CLASSIFICATION ==
def obtain_metrics(conf_m):
    fp = conf_m.sum(axis=0) - np.diag(conf_m)
    fn = conf_m.sum(axis=1) - np.diag(conf_m)
    tp = np.diag(conf_m)
    tn = conf_m.sum() - (fp + fn + tp)
    fp = fp.astype(float)
    fn = fn.astype(float)
    tp = tp.astype(float)
    tn = tn.astype(float)
    fp_n = np.nan_to_num(fp)
    fn_n = np.nan_to_num(fn)
    tp_n = np.nan_to_num(tp)
    tn_n = np.nan_to_num(tn)
    return fp_n, fn_n, tp_n, tn_n


def plotting_metrics(to_plot, results_path):
    plt.figure()
    plt.title(str(to_plot))
    plt.xlabel("epochs")
    plt.savefig(results_path + str(to_plot))
    plt.close()


def calculating_metrics(
    dataset_path, dataset_subset, dataset_name, model, results_path
):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        validation_split=0.1, rescale=1.0 / 255
    )
    dataset = datagen.flow_from_directory(
        dataset_path,
        target_size=(120, 160),
        class_mode="binary",
        shuffle=False,
        seed=0,
        subset=dataset_subset,
    )
    predictions = model.predict(dataset)
    y_pred = np.argmax(predictions, axis=1)
    y_true = dataset.classes
    conf_m = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 20))
    df_cm = pd.DataFrame(conf_m)
    sns.set(font_scale=1)  # for label size
    sns.heatmap(df_cm, annot=True, fmt="g")  # font size
    plt.savefig(results_path + dataset_name + "-confusion-matrix")
    plt.close()
    # Obtaining TP, FP, TN and FN and store it in a dictionary
    fp, fn, tp, tn = obtain_metrics(conf_m)
    far = fp / np.nan_to_num((fp + tn)) * 100
    frr = fn / np.nan_to_num((fn + tp)) * 100
    metrics = {
        "fp": fp,
        "fp_sum": sum(fp),
        "fn": fn,
        "fn_sum": sum(fn),
        "tp": tp,
        "tp_sum": sum(tp),
        "tn": tn,
        "tn_sum": sum(tn),
        "far": far,
        "far_mean": np.mean(far),
        "frr": frr,
        "frr_mean": np.mean(frr),
    }
    pkl_file = results_path + dataset_name + ".pkl"
    with open(pkl_file, "wb") as handle:
        pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return metrics


# PYTORCH FUNCTIONS
def crop_tensor(input_tensor, crop_height, crop_width):
    """
    Crop the input tensor.

    Parameters:
    - input_tensor: A 4D tensor of shape (N, C, H, W) where:
        - N is the batch size
        - C is the number of channels
        - H is the height of the input images
        - W is the width of the input images
    - crop_height: A tuple (top, bottom) specifying the number of pixels to crop from the top and bottom of the image.
    - crop_width: A tuple (left, right) specifying the number of pixels to crop from the left and right of the image.

    Returns:
    - A 4D tensor cropped to the specified dimensions.
    """
    _, _, height, width = input_tensor.shape
    top, bottom = crop_height
    left, right = crop_width

    # Calculate the indices of the crop
    start_row, end_row = top, height - bottom
    start_col, end_col = left, width - right

    # Perform the crop
    cropped_tensor = input_tensor[:, :, start_row:end_row, start_col:end_col]
    return cropped_tensor


from torch.utils.data import Dataset


class CroppedImageDataset(Dataset):
    def __init__(self, original_dataset, crop_height, crop_width):
        self.original_dataset = original_dataset
        self.crop_height = crop_height
        self.crop_width = crop_width

    def __getitem__(self, index):
        # Fetch the original data (image and label)
        image, label = self.original_dataset[index]

        # Convert PIL image to Tensor to apply cropping
        image_tensor = transforms.ToTensor()(image)

        # Crop the tensor
        cropped_tensor = crop_tensor(
            image_tensor.unsqueeze(0), self.crop_height, self.crop_width
        )  # Add batch dim for compatibility

        # Remove batch dimension after cropping
        cropped_image = cropped_tensor.squeeze(0)

        return cropped_image, label

    def __len__(self):
        return len(self.original_dataset)
