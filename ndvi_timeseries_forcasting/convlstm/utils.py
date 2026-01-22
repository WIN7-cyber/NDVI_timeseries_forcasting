import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Vars (use ALL your exported .npy files)
# Make sure these names exactly match arrays/<name>.npy
var_list = [
    "ndvi",
    "sm_30cm_mean",
    "RAIN_sum",
    "irrig_mm_sum",
    "IRRAD_sum",
    "TMIN_mean",
    "TMAX_mean",
    "VAP_mean",
    "WIND_mean",
]

# Optional (only used for plotting in some notebooks/scripts)
var_colors = {
    "ndvi": "forestgreen",
    "sm_30cm_mean": "saddlebrown",
    "RAIN_sum": "mediumblue",
    "irrig_mm_sum": "deepskyblue",
    "IRRAD_sum": "goldenrod",
    "TMIN_mean": "maroon",
    "TMAX_mean": "orangered",
    "VAP_mean": "purple",
    "WIND_mean": "slategray",
}

var_titles = {
    "ndvi": "NDVI",
    "sm_30cm_mean": "Soil moisture (30cm, mean)",
    "RAIN_sum": "Rain (sum)",
    "irrig_mm_sum": "Irrigation (mm, sum)",
    "IRRAD_sum": "Irradiance (sum)",
    "TMIN_mean": "TMIN (mean)",
    "TMAX_mean": "TMAX (mean)",
    "VAP_mean": "Vapor pressure (mean)",
    "WIND_mean": "Wind (mean)",
}


def load_data(base_path="arrays", subset=None, var_list=var_list, mask=False, auto_discover=False):
    """Load arrays from .npy files.

    Arrays are assumed to be 3D with shape (time, y, x)
    and are returned as 4D with shape (time, y, x, channel).

    Args:
        base_path (str): Folder that stores the arrays/*.npy
        subset (list|tuple|None): If provided, crops spatial dims.
            Expected as [min, max] applied to BOTH y and x: arr[:, min:max, min:max]
        var_list (list|None): Variable names (file stems) to load.
        mask (bool): True to apply mask from f"{base_path}/mask.npy" (True means masked)
        auto_discover (bool): If True, ignore var_list and load all *.npy in base_path (except mask.npy)

    Returns:
        data_arr (np.ndarray): 4D array (time, y, x, channels)
    """
    if auto_discover:
        files = [f for f in os.listdir(base_path) if f.endswith(".npy")]
        files = [f for f in files if f != "mask.npy"]
        var_list = sorted([os.path.splitext(f)[0] for f in files])

    print("Loading variables:", var_list)

    data_arrs = []

    for var in var_list:
        # Load array
        arr = np.load(os.path.join(base_path, f"{var}.npy"))

        # NDVI scaling (keep same logic as original repo)
        if var == "ndvi":
            # Convert NDVI scale from -0.3-1 to 0-1
            arr = (arr + 0.3) / 1.3

        # Crop (note: arrays are (time, y, x))
        if subset is not None:
            arr = arr[:, subset[0] : subset[1], subset[0] : subset[1]]

        # Stats
        print(var, "stats:")
        for metric, title in [
            (np.nanmean, "Mean:"),
            (np.nanstd, "Std:"),
            (np.nanmin, "Min:"),
            (np.nanmax, "Max:"),
        ]:
            print(" -", title, metric(arr))

        # Add channel dim
        data_arrs.append(np.expand_dims(arr, -1))

    # Concat channels
    data_arr = np.concatenate(data_arrs, axis=-1)
    print("Data loaded with shape:", data_arr.shape)

    if mask:
        # Load mask: shape (y, x). True means masked pixel.
        mask_arr = np.load(os.path.join(base_path, "mask.npy"))
        data_arr = np.where(mask_arr[None, :, :, None].astype(bool), np.nan, data_arr)

    return data_arr


def normalize_array(arr, mean=None, std=None):
    """Normalize array: (arr - mean) / std.

    - Supports arr as 4D (time, y, x, channel)
    - If mean/std are None, compute per-channel nanmean/nanstd over (time,y,x)

    Returns:
        normalized arr
    """
    if mean is None:
        mean = np.nanmean(arr, axis=(0, 1, 2))
    if std is None:
        std = np.nanstd(arr, axis=(0, 1, 2))
    std = np.where(std == 0, 1.0, std)
    return (arr - mean) / std


def normalize_and_split_data(inputs_all, outputs_all, train_percent=0.6, val_percent=0.2):
    """Normalize inputs & split data into training, validation, and testing sets.

    Organizes input timesteps as tmin..tmax-1 and outputs as tmin+1..tmax.

    Args:
        inputs_all (np.ndarray): 4D (time, y, x, channels)
        outputs_all (np.ndarray): 3D or 4D NDVI (time, y, x) or (time, y, x, 1)

    Returns:
        (inputs_train, outputs_train, inputs_val, outputs_val, inputs_test, outputs_test)
    """
    train_end = int(inputs_all.shape[0] * train_percent)
    val_end = train_end + int(inputs_all.shape[0] * val_percent)

    # Compute mean/std from training portion only (per channel)
    stats_arr = inputs_all[:train_end]
    stats_arr = np.reshape(
        stats_arr,
        (stats_arr.shape[0] * stats_arr.shape[1] * stats_arr.shape[2], stats_arr.shape[3]),
    )
    mean = np.nanmean(stats_arr, axis=0)
    std = np.nanstd(stats_arr, axis=0)
    std = np.where(std == 0, 1.0, std)

    inputs_train = normalize_array(inputs_all[: train_end - 1], mean, std)

    # Ensure outputs are (time, y, x, 1)
    if outputs_all.ndim == 3:
        outputs_all_4d = np.expand_dims(outputs_all, -1)
    else:
        outputs_all_4d = outputs_all

    outputs_train = outputs_all_4d[1:train_end]

    inputs_val = normalize_array(inputs_all[train_end : val_end - 1], mean, std)
    outputs_val = outputs_all_4d[train_end + 1 : val_end]

    inputs_test = normalize_array(inputs_all[val_end:-1], mean, std)
    outputs_test = outputs_all_4d[val_end + 1 :]

    print("Input train shape:", inputs_train.shape)
    print("Output train shape:", outputs_train.shape)
    print("Input val shape:", inputs_val.shape)
    print("Output val shape:", outputs_val.shape)
    print("Input test shape:", inputs_test.shape)
    print("Output test shape:", outputs_test.shape)

    return (
        inputs_train,
        outputs_train,
        inputs_val,
        outputs_val,
        inputs_test,
        outputs_test,
    )


def build_model(inputs_train, mask=None):
    input_shape = (None, *list(inputs_train.shape)[1:])
    inp = keras.layers.Input(shape=input_shape, dtype="float32", name="input")

    x = keras.layers.ConvLSTM2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        recurrent_dropout=0.25,
        data_format="channels_last",
        activation="tanh",
    )(inp)

    x = keras.layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        recurrent_dropout=0.25,
        data_format="channels_last",
        activation="tanh",
    )(x)

    out = keras.layers.Conv3D(
        filters=1,
        kernel_size=(3, 3, 3),
        activation="sigmoid",
        padding="same",
        kernel_regularizer=keras.regularizers.l2(0.01),
    )(x)

    if mask is not None:
        out = keras.layers.Multiply(name="apply_mask_output")([out, mask])

    model = keras.Model(inputs=inp, outputs=out)
    return model
