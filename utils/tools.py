"""
Utility functions for training and experiments:
1. set_seed: Set random seed for reproducibility.
2. clip_grad_norm_: Clip gradients to prevent exploding gradients.
3. save_checkpoint: Save model checkpoint.
4. load_checkpoint: Load model checkpoint.
5. count_parameters: Count the number of trainable parameters in a model.
"""

import random
import numpy as np
import torch


def set_seed(seed):
    """
    Set random seed for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



class EarlyStopping:
    """
    Early stopping to stop training when validation loss does not improve.

    It also saves the best model checkpoint.
    """

    def __init__(self, patience=3, verbose=True, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model, path):
        """
        Check if validation loss improved.
        If it improved, save the model.
        """

        score = -val_loss   ### siccome vuoi sempre una val.loss minore, facendo cosi una loss piu piccola diventa uno score piu grande

        if self.best_score is None: ### prima volta che viene chiamata la funzione, quindi non c'è ancora uno score migliore
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)

        elif score < self.best_score + self.delta: ### se lo score attuale è peggiore dello score migliore + delta, allora non c'è miglioramento
            self.counter += 1

            if self.verbose: ### stampa il numero di volte consecutive che la loss non è migliorata
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience: ### se il numero di volte consecutive che la loss non è migliorata supera patience, allora ferma il training
                self.early_stop = True

        else: ### se lo score attuale è migliore dello score migliore + delta, allora c'è miglioramento
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """
        Save model checkpoint when validation loss decreases.
        """

        if self.verbose:
            print(
                f"Validation loss decreased "
                f"({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model..."
            )

        torch.save(model.state_dict(), path + "/checkpoint.pth") ### salva i pesi del modello MIGLIORE in un file checkpoint.pth nella cartella path
        self.val_loss_min = val_loss



def adjust_learning_rate(optimizer, epoch, learning_rate, lradj="type1"):
    """
    Adjust learning rate during training.

    This follows the learning rate schedules used in the official Autoformer code.
    """

    if lradj == "type1":
        lr_adjust = {
            epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))
        }

    elif lradj == "type2":
        lr_adjust = {
            2: 5e-5,
            4: 1e-5,
            6: 5e-6,
            8: 1e-6,
            10: 5e-7,
            15: 1e-7,
            20: 5e-8
        }

    else:
        lr_adjust = {}

    if epoch in lr_adjust:
        lr = lr_adjust[epoch]

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        print(f"Updating learning rate to {lr}")

        return lr

    return optimizer.param_groups[0]["lr"]