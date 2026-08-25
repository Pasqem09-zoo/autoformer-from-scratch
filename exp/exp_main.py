"""
Main experiment class for Autoformer.

This file contains the training, validation and test logic.
"""

import os
import time

import torch
import torch.nn as nn
import numpy as np
import wandb

from utils.metrics import metric
from models.autoformer import Autoformer
from data_provider.data_loader import get_data_loader
from utils.tools import EarlyStopping, adjust_learning_rate


class ExpMain:
    """
    Main experiment class.

    It builds:
    - model
    - data loaders
    - optimizer
    - loss function

    Then it manages:
    - training
    - validation
    - testing
    """

    def __init__(self, config):
        self.config = config

        self.device = self._get_device()
        self.model = self._build_model().to(self.device) ### creiamo Autoformer e lo mandiamo su mps, cuda o cpu

        self.criterion = self._select_criterion()
        self.optimizer = self._select_optimizer()

    def _get_device(self):
        """
        Select the best available device.
        """

        if self.config.get("use_cuda", True) and torch.cuda.is_available():
            device = "cuda"

        elif self.config.get("use_mps", True) and torch.backends.mps.is_available():
            device = "mps"

        else:
            device = "cpu"

        print(f"Using device: {device}")

        return device

    def _build_model(self):
        """
        Build Autoformer model.
        """

        model = Autoformer(self.config)

        return model

    def _select_criterion(self):
        """
        Select loss function.

        Autoformer is trained using Mean Squared Error.
        """

        criterion = nn.MSELoss()

        return criterion

    def _select_optimizer(self):
        """
        Select optimizer.

        The official Autoformer code uses Adam.
        """

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"]
        )

        return optimizer

    def _get_data(self, flag):
        """
        Create dataset and dataloader.

        flag can be:
        - "train"
        - "val"
        - "test"
        """

        if flag == "train": ### shuffle=True solo per il training. Validation e test invece devono restare ordinati, così la valutazione è più pulita e riproducibile
            shuffle = True
        else:
            shuffle = False

        dataset, data_loader = get_data_loader(
            data_path=self.config["data_path"],
            flag=flag,
            seq_len=self.config["seq_len"],
            label_len=self.config["label_len"],
            pred_len=self.config["pred_len"],
            features=self.config["features"],
            target=self.config["target"],
            batch_size=self.config["batch_size"],
            shuffle=shuffle,
            scale=True
        )

        print(f"{flag} dataset size: {len(dataset)}")

        return dataset, data_loader


    def _predict(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        """
        Build decoder input and run the model.

        The decoder input is made of:
        - the first label_len points of batch_y
        - pred_len zeros for the future part
        """

        dec_zeros = torch.zeros_like(
            batch_y[:, -self.config["pred_len"]:, :]
        ).float()

        dec_inp = torch.cat(
            [
                batch_y[:, :self.config["label_len"], :],
                dec_zeros
            ],
            dim=1
        ).float().to(self.device)

        output = self.model(
            batch_x,
            batch_x_mark,
            dec_inp,
            batch_y_mark
        )
        target = batch_y[:, -self.config["pred_len"]:, :]

        return output, target
    

    def validate(self, val_loader):
        """
        Evaluate the model on validation data.
        """

        self.model.eval() ### mette il modello in modalità evaluation

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad(): ### dice a PyTorch: “non salvare il grafo computazionale e non calcolare gradienti”
            for batch in val_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_x_mark = batch_x_mark.to(self.device)
                batch_y_mark = batch_y_mark.to(self.device)

                ### la loss va calcolata solo sulla parte futura vera, cioè sugli ultimi pred_len punti
                output, target = self._predict(
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark
                )

                loss = self.criterion(output, target)
                total_loss += loss.item()
                num_batches += 1

        average_loss = total_loss / num_batches

        self.model.train()

        return average_loss ### in pratica questa funzione restituisce la loss media sui batch di validation. Serve per capire se il modello sta migliorando o peggiorando durante il training


    def train(self):
        """
        Train the model.
        cuore del file: prende i dati, fa epoche di training, calcola validation loss, usa early stopping, salva il checkpoint migliore e aggiorna il learning rate.
        """

        train_dataset, train_loader = self._get_data("train") ### otteniamo il dataset e il dataloader per il training
        val_dataset, val_loader = self._get_data("val")

        checkpoint_dir = self.config.get("checkpoint_dir", "checkpoints") ### directory dove salvare i pesi del modello. Se non è specificata nel config, usa "checkpoints" come default
        os.makedirs(checkpoint_dir, exist_ok=True)

        early_stopping = EarlyStopping(
            patience=self.config["patience"],
            verbose=True
        )

        train_epochs = self.config["epochs"]
        train_steps = len(train_loader)
        time_now = time.time()

        for epoch in range(1, train_epochs + 1): ### questo ciclo fa le epoche di training
            start_time = time.time() ### serve per misurare quanto tempo impiega un'epoca di training
            time_now = time.time()
            
            self.model.train()

            total_train_loss = 0.0
            num_batches = 0
            iter_count = 0

            for batch_idx, batch in enumerate(train_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_x_mark = batch_x_mark.to(self.device)
                batch_y_mark = batch_y_mark.to(self.device)

                self.optimizer.zero_grad()

                output, target = self._predict( ### ### [B, pred_len, c_out]
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark
                )

                loss = self.criterion(output, target)
                loss.backward()

                if self.config.get("use_grad_clip", False): ### TODO: NEL CODICE DEGLI AUTORI NON C'è
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.get("grad_clip", 1.0)
                    )

                self.optimizer.step()

                total_train_loss += loss.item() ### loss media sui batch di training
                num_batches += 1
                iter_count += 1

                if (batch_idx + 1) % 100 == 0: ### ogni 100 batch stampiamo la loss media sui batch di training e stimiamo quanto tempo manca alla fine del training
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            batch_idx + 1,
                            epoch,
                            loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((train_epochs - epoch) * train_steps + (train_steps - batch_idx - 1))
                    
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed,
                            left_time
                        )
                    )

                    if self.config.get("wandb_enabled", False):
                        global_step = (epoch - 1) * train_steps + batch_idx + 1

                        wandb.log({
                            "batch_loss_every_100": loss.item(),
                            "epoch": epoch,
                            "batch_idx": batch_idx + 1,
                            "global_step": global_step,
                            "learning_rate": self.optimizer.param_groups[0]["lr"]
                        })

                    iter_count = 0
                    time_now = time.time()

            train_loss = total_train_loss / num_batches ### loss media sui batch di training di 1 epoca
            val_loss = self.validate(val_loader) ### loss media sui batch di validation di 1 epoca

            epoch_time = time.time() - start_time

            print( ### statistiche di training e validation per 1 epoca
                f"Epoch {epoch}/{train_epochs} | "
                f"train loss: {train_loss:.6f} | "
                f"val loss: {val_loss:.6f} | "
                f"time: {epoch_time:.2f}s"
            )

            # WANDB
            if self.config.get("wandb_enabled", False):
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "epoch_time": epoch_time,
                    "learning_rate": self.optimizer.param_groups[0]["lr"]
                })

            early_stopping( ### “guarda la validation loss appena ottenuta; se è la migliore finora, salva il modello”; se la val.loss non migliora per "patience" epoche consecutive, ferma il training
                val_loss=val_loss,
                model=self.model,
                path=checkpoint_dir
            )

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate( ### aggiorna il lr secondo la strategia scelta (type1, type2, ecc)
                optimizer=self.optimizer,
                epoch=epoch,
                learning_rate=self.config["learning_rate"],
                lradj=self.config.get("lradj", "type1")
            )

        ### “ok, il training è finito; ricarico i pesi migliori, non necessariamente quelli dell’ultima epoca”
        best_model_path = os.path.join(checkpoint_dir, "checkpoint.pth") ### salva i pesi del modello migliore (cioè quello con la val.loss più bassa) in checkpoint.pth***
        self.model.load_state_dict(
            torch.load(best_model_path, map_location=self.device)
        )
        ### *** epoca 1
            # training su tutti i batch
            # → calcoliamo train_loss media

            # validation su tutti i batch di validation
            # → calcoliamo val_loss media

            # EarlyStopping guarda questa val_loss
            # → se è la migliore finora, salva checkpoint.pth

        return self.model


    def test(self, load_checkpoint=True):
        """
        Test the model on the test set.

        It computes:
        - MAE
        - MSE

        carica il best checkpoint
        scorre il test_loader
        fa predizioni
        salva predizioni e target veri
        calcola MAE e MSE
        stampa i risultati
        salva tutto in results/
        """

        test_dataset, test_loader = self._get_data("test")

        if load_checkpoint:
            checkpoint_dir = self.config.get("checkpoint_dir", "checkpoints")
            best_model_path = os.path.join(checkpoint_dir, "checkpoint.pth")

            print(f"Loading best model from: {best_model_path}")

            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )

        preds = []
        trues = []

        self.model.eval()

        with torch.no_grad():
            for batch in test_loader: ### questo ciclo scorre tutti i batch del test_loader, fa predizioni e salva le predizioni e i target veri in due liste separate
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_x_mark = batch_x_mark.to(self.device)
                batch_y_mark = batch_y_mark.to(self.device)

                output, target = self._predict( ### anche nel test usiamo lo stesso identico meccanismo degli autori: decoder input con parte nota + zeri futuri
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark
                )

                pred = output.detach().cpu().numpy()
                true = target.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        print("test prediction shape:", preds.shape)
        print("test true shape:", trues.shape)

        mae_score, mse_score = metric(preds, trues)

        print(f"Test MSE: {mse_score:.6f}")
        print(f"Test MAE: {mae_score:.6f}")

        # WANDB
        if self.config.get("wandb_enabled", False):
            wandb.log({
                "test_mse": mse_score,
                "test_mae": mae_score
            })

        results_dir = self.config.get("results_dir", "results") ### chiamo la directory dove salvare i risultati il cui nome è nel config, altrimenti uso "results" come default
        os.makedirs(results_dir, exist_ok=True)

        np.save(os.path.join(results_dir, "pred.npy"), preds)
        np.save(os.path.join(results_dir, "true.npy"), trues)
        np.save(
            os.path.join(results_dir, "metrics.npy"),
            np.array([mae_score, mse_score])
        )

        return mae_score, mse_score