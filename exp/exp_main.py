"""
Main experiment class for Autoformer.

This file contains the training, validation and test logic.
It builds the model, the data loaders, the optimizer and the loss function.
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
from utils.experiment_summary import save_experiment_summary


class ExpMain:

    def __init__(self, config):
        self.config = config

        self.device = self._get_device()
        self.model = self._build_model().to(self.device)

        self.criterion = self._select_criterion()
        self.optimizer = self._select_optimizer()

        self.epoch_history = []
        self.train_size = None
        self.val_size = None
        self.test_size = None
        self.train_steps = None
        self.early_stopped = False

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

        if flag == "train":
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
            dataset=self.config["dataset"],
            freq=self.config["freq"]
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

        output = self.model( # call Autoformer model and get the predictions
            batch_x,
            batch_x_mark,
            dec_inp,
            batch_y_mark
        )
        target = batch_y[:, -self.config["pred_len"]:, :]

        return output, target
    

    def validate(self, val_loader):
        """
        Evaluate the model on validation set and return the average loss.
        """

        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad(): # no gradient calculation during validation
            for batch in val_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_x_mark = batch_x_mark.to(self.device)
                batch_y_mark = batch_y_mark.to(self.device)

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

        return average_loss


    def train(self):
        """
        Train the model.

        It takes care of:
        - training epochs
        - validation loss calculation
        - early stopping
        - saving the best checkpoint
        - learning rate adjustment
        """

        train_dataset, train_loader = self._get_data("train")
        val_dataset, val_loader = self._get_data("val")
        self.train_size = len(train_dataset)
        self.val_size = len(val_dataset)
        self.train_steps = len(train_loader)

        checkpoint_dir = self.config.get("checkpoint_dir", "checkpoints") # directory where to save model weights. If not specified in config, use "checkpoints" as default
        os.makedirs(checkpoint_dir, exist_ok=True)

        early_stopping = EarlyStopping(
            patience=self.config["patience"],
            verbose=True
        )

        train_epochs = self.config["epochs"]
        train_steps = self.train_steps

        # saving the experiment summary will be done in the test() method, after the test is completed
        self.epoch_history = []

        for epoch in range(1, train_epochs + 1): # loop over epochs
            start_time = time.time() # start time of the epoch
            time_now = time.time() # start time of the batch iteration
            
            self.model.train()

            total_train_loss = 0.0
            num_batches = 0
            iter_count = 0

            for batch_idx, batch in enumerate(train_loader): # loop over batches in the Dataloader training set: batch_idx is the index of the batch, batch is a tuple containing (batch_x, batch_y, batch_x_mark, batch_y_mark)
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_x_mark = batch_x_mark.to(self.device)
                batch_y_mark = batch_y_mark.to(self.device)

                self.optimizer.zero_grad()

                output, target = self._predict( # [B, pred_len, c]
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark
                )

                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item() # loss of the current batch in the current epoch
                num_batches += 1
                iter_count += 1

                if (batch_idx + 1) % 100 == 0: # every 100 batches print the loss of the last batch and estimate how much time is left until the end of training
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
                        epoch_progress = epoch + (batch_idx + 1) / train_steps
                        wandb.log({
                            "epoch_progress": epoch_progress,
                            "batch_loss_every_100": loss.item()
                        })

                    iter_count = 0
                    time_now = time.time() # end of the batch iteration, reset the timer for the next 100 batches

                    if self.config.get("wandb_enabled", False):
                        epoch_progress = epoch + (batch_idx + 1) / train_steps
                        wandb.log({
                            "epoch_progress": epoch_progress,
                            "batch_loss_every_100": loss.item()
                        })

                    iter_count = 0
                    time_now = time.time()

            train_loss = total_train_loss / num_batches # average loss over all batches in the current epoch
            val_loss = self.validate(val_loader) # average loss over all batches in the validation set, in the current epoch

            epoch_time = time.time() - start_time

            print(
                f"Epoch {epoch}/{train_epochs} | "
                f"train loss: {train_loss:.6f} | "
                f"val loss: {val_loss:.6f} | "
                f"time: {epoch_time:.2f}s"
            )

            self.epoch_history.append({ # save epoch history for later analysis and for saving the experiment summary at the end of training
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch_time": epoch_time,
                "learning_rate": self.optimizer.param_groups[0]["lr"]
            })

            # WANDB
            if self.config.get("wandb_enabled", False):
                wandb.log({
                        "epoch": epoch,
                        "epoch_value": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "epoch_time": epoch_time,
                        "learning_rate": self.optimizer.param_groups[0]["lr"]
                })

            early_stopping(
                val_loss=val_loss,
                model=self.model,
                path=checkpoint_dir
            )

            if early_stopping.early_stop:
                print("Early stopping")
                self.early_stopped = True
                break

            adjust_learning_rate(   # adjust the learning rate according to the schedule specified in the config file
                optimizer=self.optimizer,
                epoch=epoch,
                learning_rate=self.config["learning_rate"],
                lradj=self.config.get("lradj", "type1")
            )

        best_model_path = os.path.join(checkpoint_dir, "checkpoint.pth")    # checkpoint.pth contains the best model weights***
        self.model.load_state_dict(     # load the best model weights after training is complete, not necessarily the weights from the last epoch
            torch.load(best_model_path, map_location=self.device)
        )
        # *** epoch 1
            # training on all batches
            # → calculate average train_loss

            # validation on all validation batches
            # → calculate average val_loss

            # EarlyStopping looks at this val_loss
            # → if it's the best so far, save checkpoint.pth

        return self.model


    def test(self, load_checkpoint=True):
        """
        Test the model on the test set.

        It computes:
        - MAE
        - MSE
        """

        test_dataset, test_loader = self._get_data("test")
        self.test_size = len(test_dataset)

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
            for batch in test_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_x_mark = batch_x_mark.to(self.device)
                batch_y_mark = batch_y_mark.to(self.device)

                output, target = self._predict(
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

        results_dir = self.config.get("results_dir", "results") # directory where to save the test results. If not specified in config, use "results" as default
        os.makedirs(results_dir, exist_ok=True)
        np.save(
            os.path.join(results_dir, "metrics.npy"),
            np.array([mae_score, mse_score])
        )
        if self.config.get("save_predictions", False):
            np.save(os.path.join(results_dir, "pred.npy"), preds)
            np.save(os.path.join(results_dir, "true.npy"), trues)

        save_experiment_summary(
            config=self.config,
            epoch_history=self.epoch_history,
            mae_score=mae_score,
            mse_score=mse_score,
            device=self.device,
            train_size=self.train_size,
            val_size=self.val_size,
            test_size=self.test_size,
            train_steps=self.train_steps,
            early_stopped=self.early_stopped
        )

        return mae_score, mse_score