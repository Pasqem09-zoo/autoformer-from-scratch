"""
Dataset and DataLoader utilities for time series forecasting.
"""

import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from utils.timefeatures import time_features


class ETTDataset(Dataset):
    """
    PyTorch Dataset for ETT time series forecasting.

    It builds sliding windows for Autoformer.
    """

    def __init__(
        self,
        data_path, ### percorso del csv tipo data/raw/ETTm2.csv
        flag="train", ### train or test or val
        seq_len=96, ### lunghezza della finestra temporale passata data all'encoder
        label_len=48, ### lunghezza della finestra temporale passata data al decoder
        pred_len=96, ### lunghezza della finestra temporale da predire
        features="M", ### M: multivariate, S: univariate
        target="OT", ### colonna target da predire
        scale=True ### se normalizzare o meno i dati
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
        """

        df_raw = pd.read_csv(self.data_path)

        # The ETT datasets have a date column and several numerical columns.
        # Example columns:
        # date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT

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
        ### Nei dataset di forecasting non facciamo split casuale, perché altrimenti mischieremmo passato e futuro
        ### Official ETTm2 split used by the Autoformer authors.
        ### ETTm2 is sampled every 15 minutes, so there are 4 observations per hour.
        ### train = 12 months, validation = 4 months, test = 4 months
        num_train = 12 * 30 * 24 * 4 ### 12 mesi × 30 giorni × 24 ore × 4 punti per ora
        num_val = 4 * 30 * 24 * 4 ### 4 mesi × 30 giorni × 24 ore × 4 punti per ora
        num_test = 4 * 30 * 24 * 4 ### 4 mesi × 30 giorni × 24 ore × 4 punti per ora


        ### per ogni variabile (colonna) calcola media e sd e normalizza tutti i dati (train,test e valid) per avere tutto sulla stessa scala
        train_data = data[0:num_train] ### RICORDA: la posizione num_train è esclusa
        if self.scale: ### nell'init sta a true
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)


        ### La colonna date nel CSV viene letta inizialmente come testo. Con pd.todate() la trasformiamo in una vera data pandas, così possiamo estrarre mese, giorno, ora, ecc
        df_stamp = df_raw[["date"]].copy()
        df_stamp["date"] = pd.to_datetime(df_stamp["date"])  ### trasforma in una vera data pandas

        data_stamp = time_features(df_stamp["date"].values, freq="t")


        border1s = { ### è il punto da cui iniziamo a prendere i dati per costruire le finestre di train val e test
            "train": 0,
            "val": num_train - self.seq_len, ### la validation sborda un pochino indietro perche la prima finestra di validation ha bisogno anche del passato immediatamente precedente, cioè della fine del train (gli ultimi seq_len punti del train)
            "test": num_train + num_val - self.seq_len
        }

        border2s = { ### è il punto dove finisce la porzione di dati di border1s
            "train": num_train,
            "val": num_train + num_val,
            "test": num_train + num_val + num_test
        }

        border1 = border1s[self.flag]
        border2 = border2s[self.flag]

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        ### train: data[0 : num_train]
        ### validation: data[num_train - seq_len : num_train + num_val]
        ### test: data[num_train + num_val - seq_len : num_train + num_val + num_test]
        ### graficamente le finestre sono i pezzi che autoformer prende per fare le predizioni:
        # serie lunga:
        # |--------------------------------------------------------------------------------|
        # finestra 1:
        # |---- passato 96 ----|---- futuro 96 ----|
        # finestra 2:
        #    |---- passato 96 ----|---- futuro 96 ----|
        # finestra 3:
        #       |---- passato 96 ----|---- futuro 96 ----|

        self.data_stamp = data_stamp[border1:border2]



    ### dice quante finestre possiamo costruire dentro il pezzo di serie che ho selezionato
    def __len__(self):
        """
        Return the number of sliding windows available in the dataset.
        """

        return len(self.data_x) - self.seq_len - self.pred_len + 1



    ### è ciò che viene attivato quando chiami un esempio del dataset: dato un indice costruisce la finestra index
    ### e.g. se index=0 lui costruisce la prima finestra:
    ### seq_x = passato dato all'encoder
    ### seq_y = ultimi label_len punti noti + pred_len punti futuri veri
    ### seq_x_mark = time features associate a seq_x
    ### seq_y_mark = time features associate a seq_y
    ###
    ### con i parametri base seq_len = 96, label_len = 48, pred_len = 96 sarà:
    ### seq_x = data[0:96]
    ### seq_y = data[48:192]
    ###
    ### quindi seq_y contiene:
    ### data[48:96]   -> parte nota data al decoder
    ### data[96:192]  -> futuro vero da prevedere
    ###
    ### nel training loop useremo solo la parte finale di seq_y come target:
    ### target = seq_y[:, -pred_len:, :]
    ###
    ### QUINDI GETITEM TRASFORMA UNA LUNGA SERIE TEMPORALE IN UN ESEMPIO SUPERVISIONATO
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
            torch.tensor(seq_y, dtype=torch.float32), ### contiene ultimi label_len punti noti + pred_len punti futuri veri
            torch.tensor(seq_x_mark, dtype=torch.float32),
            torch.tensor(seq_y_mark, dtype=torch.float32)
        )
    ### il batch è:
    # batch_x      → input encoder
    # batch_y      → decoder input completo + target futuro
    # batch_x_mark → time features encoder
    # batch_y_mark → time features decoder



    ### Durante il training lavoriamo sui dati normalizzati, perché la rete impara meglio.
    ### Però alla fine, quando faremo previsioni e magari vorremo confrontare graficamente le previsioni con i dati reali, vogliamo riportare tutto alla scala originale
    def inverse_transform(self, data):
        """
        Transform normalized data back to the original scale.
        """

        return self.scaler.inverse_transform(data)




class CustomDataset(Dataset):
    """
    Dataset class for non-ETT datasets such as Electricity, Traffic and Weather.

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
        self.freq = freq

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
            set_type = 0
        elif self.flag == "val":
            set_type = 1
        else:
            set_type = 2

        border1 = border1s[set_type]
        border2 = border2s[set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]
        else:
            raise ValueError("features must be one of: S, M, MS")

        if self.scale:
            train_data = df_data.iloc[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]].iloc[border1:border2].copy()
        df_stamp["date"] = pd.to_datetime(df_stamp["date"])

        data_stamp = time_features(df_stamp["date"].values, freq=self.freq)

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

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size, ### quante finestre voglio prendere in un batch
        shuffle=shuffle_flag, ### nel training voglio mischiare le finestre, nel test e validation no perché voglio vedere come il modello si comporta su finestre consecutive della serie temporale
        num_workers=0,
        drop_last=True  ### se l ultimo batch non è completo (cioè non contiene batch_size finestre) lo scarta. Serve per evitare problemi con il batchnorm
    )

    return dataset, data_loader