"""
Dataset and DataLoader utilities for time series forecasting.
"""

import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


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

        num_train = int(len(data) * 0.7)
        num_val = int(len(data) * 0.1)
        ### Nei dataset di forecasting non facciamo split casuale, perché altrimenti mischieremmo passato e futuro. Sarebbe come studiare per l’esame guardando già le risposte del compito: il modello “bara”
        num_test = len(data) - num_train - num_val


        ### per ogni variabile (colonna) calcola media e sd e normalizza tutti i dati (train,test e valid) per avere tutto sulla stessa scala
        train_data = data[0:num_train] ### RICORDA: la posizione num_train è esclusa
        if self.scale: ### nell'init sta a true
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)


        ### La colonna date nel CSV viene letta inizialmente come testo. Con pd.todate() la trasformiamo in una vera data pandas, così possiamo estrarre mese, giorno, ora, ecc
        df_stamp = df_raw[["date"]]
        df_stamp["date"] = pd.to_datetime(df_stamp["date"])

        df_stamp["month"] = df_stamp["date"].dt.month
        df_stamp["day"] = df_stamp["date"].dt.day
        df_stamp["weekday"] = df_stamp["date"].dt.weekday
        df_stamp["hour"] = df_stamp["date"].dt.hour
        df_stamp["minute"] = df_stamp["date"].dt.minute

        data_stamp = df_stamp[["month", "day", "weekday", "hour", "minute"]].values


        border1s = { ### è il punto da cui iniziamo a prendere i dati per costruire le finestre di train val e test
            "train": 0,
            "val": num_train - self.seq_len, ### la validation sborda un pochino indietro perche la prima finestra di validation ha bisogno anche del passato immediatamente precedente, cioè della fine del train (gli ultimi seq_len punti del train)
            "test": num_train + num_val - self.seq_len
        }

        border2s = { ### è il punto dove finisce la porzione di dati di border1s
            "train": num_train,
            "val": num_train + num_val,
            "test": len(data)
        }

        border1 = border1s[self.flag]
        border2 = border2s[self.flag]

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        ### train: data[0 : num_train]
        ### validation: data[num_train - seq_len : num_train + num_val]
        ### test: data[num_train + num_val - seq_len : fine]
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


    ### è cio che viene attivato quando chiami un esempio del dataset: dato un indice costruisce la finestra index
    ### e.g. se index=0 lui costruisce la prima finestra: x_enc = passato dato all'encoder,x_dec = input dato al decoder,
    # y = futuro vero da prevedere. con i parametri base seq_len = 96,label_len = 48,pred_len = 96 sarà:
    # x_enc = data[0:96], x_dec = data[48:192], y=data[96:192]
    # QUINDI GETITEM TRASFORMA UNA LUNGA SERIE TEMPORALE IN UN ESEMPIO SPERUVISIONATO
    def __getitem__(self, index):
        """
        Build one sliding window.
        """

        s_begin = index
        s_end = s_begin + self.seq_len

        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        x_enc = self.data_x[s_begin:s_end] ### passato dato al modello
        y = self.data_y[s_end:r_end] ### contiene il futuro vero da prevedere

        x_mark_enc = self.data_stamp[s_begin:s_end]
        x_mark_dec = self.data_stamp[r_begin:r_end]

        x_dec = self.data_x[r_begin:r_end] ### contiene ultimi label_len punti (noti) + pred_len punti futuri

        return (
            torch.tensor(x_enc, dtype=torch.float32),
            torch.tensor(x_mark_enc, dtype=torch.float32),
            torch.tensor(x_dec, dtype=torch.float32),
            torch.tensor(x_mark_dec, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )




def get_data_loader(
    data_path,
    flag,
    seq_len,
    label_len,
    pred_len,
    features,
    target,
    batch_size, ### quante finestre voglio prendere in un batch
    shuffle=True, ### nel training voglio mischiare le finestre, nel test e validation no perché voglio vedere come il modello si comporta su finestre consecutive della serie temporale
    scale=True
):
    """
    Create Dataset and DataLoader for ETT data.
    """

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

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True ### se l ultimo batch non è completo (cioè non contiene batch_size finestre) lo scarta. Serve per evitare problemi con il batchnorm
    )

    return dataset, data_loader