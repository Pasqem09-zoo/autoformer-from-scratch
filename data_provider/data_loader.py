"""
Dataset and DataLoader utilities for time series forecasting.

This module provides classes and functions to load and preprocess time series data for training and evaluating forecasting models.

Classes:
- ETTDataset: Dataset class for the ETTm2 dataset.
- CustomDataset: Dataset class for other datasets such as Electricity, Traffic, and Weather.
"""

import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from utils.timefeatures import time_features


class ETTDataset(Dataset):

    def __init__(
        self,
        data_path,
        flag="train",    # train or test or val
        seq_len=96,      # past length given to the encoder
        label_len=48,    # length of the known part of the decoder input
        pred_len=96,     # length of the prediction window
        features="M",    # M: multivariate, S: univariate
        target="OT",     # column to predict
        scale=True       # whether to normalize the data
    ):
        super().__init__()

        self.data_path = data_path
        self.flag = flag

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        self.features = features
        self.target = target
        self.scale = scale

        self.scaler = StandardScaler()

        self._read_data()


    def _read_data(self):
        """
        Read the CSV file, split the data and select the columns.

        The ETT datasets have a date column and several numerical columns:
        date, HUFL, HULL, MUFL, MULL, LUFL, LUFL, LULL, OT
        """

        df_raw = pd.read_csv(self.data_path)

        if self.features == "M":
            # Multivariate forecasting: use all variables except date
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]

        elif self.features == "S":
            # Univariate forecasting: use only the target column
            df_data = df_raw[[self.target]]

        else:
            raise ValueError("features must be either 'M' or 'S'")


        data = df_data.values
        # Official ETTm2 split used by the Autoformer authors.
        # ETTm2 is sampled every 15 minutes, so there are 4 observations per hour.
        # train = 12 months, validation = 4 months, test = 4 months
        
        num_train = 12 * 30 * 24 * 4    # 12 months × 30 days × 24 hours × 4 points per hour
        num_val = 4 * 30 * 24 * 4       # 4 months × 30 days × 24 hours × 4 points per hour
        num_test = 4 * 30 * 24 * 4      # 4 months × 30 days × 24 hours × 4 points per hour


        # normalization is done only on the training data and then applied to the entire dataset
        train_data = data[0:num_train]
        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)


        df_stamp = df_raw[["date"]].copy()
        df_stamp["date"] = pd.to_datetime(df_stamp["date"])             # convert the date column to pandas datetime format
        data_stamp = time_features(df_stamp["date"].values, freq="t")   # set default frequency to 15 minutes (t = 15min) for ETTm2 dataset


        border1s = {    # the starting index of the data for training, validation, and testing
            "train": 0,
            "val": num_train - self.seq_len,
            "test": num_train + num_val - self.seq_len
        }

        border2s = {
            "train": num_train,
            "val": num_train + num_val,
            "test": num_train + num_val + num_test
        }

        border1 = border1s[self.flag]
        border2 = border2s[self.flag]

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp[border1:border2]



    def __len__(self):
        """
        Return the number of sliding windows available in the dataset.
        """

        return len(self.data_x) - self.seq_len - self.pred_len + 1




    def __getitem__(self, index):
        """
        Build one sliding window.
        """

        s_begin = index
        s_end = s_begin + self.seq_len

        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return (
            torch.tensor(seq_x, dtype=torch.float32),
            torch.tensor(seq_y, dtype=torch.float32),
            torch.tensor(seq_x_mark, dtype=torch.float32),
            torch.tensor(seq_y_mark, dtype=torch.float32)
        )



    def inverse_transform(self, data):
        """
        Transform normalized data back to the original scale.
        """

        return self.scaler.inverse_transform(data)




class CustomDataset(Dataset):
    """
    Dataset class for non-ETT datasets: Electricity, Traffic, Weather, Exchange, and ILI.

    This class follows the official Autoformer split:
        70% train
        10% validation
        20% test
    """

    def __init__(
        self,
        data_path,
        flag,
        seq_len,
        label_len,
        pred_len,
        features="M",
        target="OT",
        scale=True,
        freq="h"
    ):
        super().__init__()

        assert flag in ["train", "val", "test"]

        self.data_path = data_path
        self.flag = flag

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        self.features = features
        self.target = target
        self.scale = scale
        self.freq = freq        # frequency of the time series data (e.g., 'h' for hourly, 'd' for daily, etc.)

        self.scaler = StandardScaler()

        self.__read_data__()



    def __read_data__(self):
        df_raw = pd.read_csv(self.data_path)

        # The dataframe must contain a date column.
        # The target column is moved to the end, as in the official implementation.
        cols = list(df_raw.columns)
        cols.remove("date")
        cols.remove(self.target)

        df_raw = df_raw[["date"] + cols + [self.target]]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_val = len(df_raw) - num_train - num_test

        border1s = [
            0,
            num_train - self.seq_len,
            len(df_raw) - num_test - self.seq_len
        ]

        border2s = [
            num_train,
            num_train + num_val,
            len(df_raw)
        ]

        if self.flag == "train":
            set_type = 0            # set_type is used to select the appropriate borders for train, val, and test splits
        elif self.flag == "val":
            set_type = 1
        else:
            set_type = 2

        border1 = border1s[set_type]
        border2 = border2s[set_type]

        if self.features == "M" or self.features == "MS":                       # Multivariate forecasting: use all variables except date
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":                                              # Univariate forecasting: use only the target column
            df_data = df_raw[[self.target]]
        else:
            raise ValueError("features must be one of: S, M, MS")

        if self.scale:
            train_data = df_data.iloc[border1s[0]:border2s[0]]                  # Use only the training data to fit the scaler
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]].iloc[border1:border2].copy()
        df_stamp["date"] = pd.to_datetime(df_stamp["date"])
        data_stamp = time_features(df_stamp["date"].values, freq=self.freq)     # Generate time features for the timestamp

        self.data_x = data[border1:border2].astype("float32")
        self.data_y = data[border1:border2].astype("float32")
        self.data_stamp = data_stamp.astype("float32")



    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark



    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1



    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)




def get_data_loader(
    data_path,
    flag,
    seq_len,
    label_len,
    pred_len,
    features,
    target,
    batch_size,
    dataset="ETTm2",
    freq="h",
    scale=True
):
    """
    Create dataset and dataloader.

    This function supports both ETTm2 and custom datasets.
    """

    dataset_name = dataset.lower()

    if dataset_name == "ettm2":
        dataset = ETTDataset(
            data_path=data_path,
            flag=flag,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            features=features,
            target=target,
            scale=scale
        )

    elif dataset_name in ["electricity", "traffic", "weather", "exchange", "ili"]:
        dataset = CustomDataset(
            data_path=data_path,
            flag=flag,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            features=features,
            target=target,
            scale=scale,
            freq=freq
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    shuffle_flag = True if flag == "train" else False

    data_loader = DataLoader(   # DataLoader is a PyTorch utility that provides an iterable over the dataset, allowing for easy batching and shuffling of data.
        dataset,
        batch_size=batch_size,  # number of sliding windows in each batch
        shuffle=shuffle_flag,   # shuffle the data only for training
        num_workers=0,
        drop_last=True          # drop the last incomplete batch to avoid issues with batch normalization
    )

    return dataset, data_loader