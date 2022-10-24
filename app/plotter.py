import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import ConfusionMatrixDisplay


def plot_cm(y_pred: torch.Tensor, y_true: torch.Tensor, class_labels: list[str]) -> None:
    """
    Plots a confusion matrix based on a given confusion matrix torch.Tensor.

    :param y_pred: (torch.Tensor) model predictions
    :param y_true: (torch.Tensor) image labels
    :param class_labels: (list[string]) a list of class label names
    """
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_labels, cmap=plt.cm.Blues)
    plt.show()


def plot_losses(train_losses: list[float], valid_losses: list[float]) -> None:
    """Plot the model training and validation losses against the amount of epoch iterations.

     :param train_losses: (list[float]) a list of training losses per training epoch
     :param valid_losses: (list[float]) a list of validation losses per training epoch
     """
    fig = plt.figure()
    epochs = range(len(train_losses))
    plt.plot(epochs, train_losses, label='training loss')
    plt.plot(epochs, valid_losses, label='validation loss')
    plt.xlabel("Iterations")
    plt.ylabel("Losses")
    plt.legend(loc="upper right")
    plt.title(f"Loss Comparison")
    plt.show()


@torch.no_grad()
def plot_model_preds(model: models, test_data: DataLoader, class_labels: list[str]) -> None:
    """
    Plots a set of image classification predictions. Images are labelled with the prediction and the true label in
    brackets. Green names indicate a correct prediction, and red incorrect.

    :param model: (torchvision.models) the model to make the predictions
    :param test_data: (torch.utils.data.DataLoader) the test data to make the predictions
    :param class_labels: (list[string]) a list of class label names
    """
    n_cols = 10
    n_rows = 2

    # Get a single batch of test images
    dataiter = iter(test_data)
    imgs, lbls = dataiter.next()
    imgs, lbls = imgs[:n_cols * n_rows], lbls[:n_cols * n_rows]

    # Get predictions
    model = model.cpu()
    preds = torch.exp(model.forward(imgs)).max(dim=1)[1].numpy()

    # Plot n_cols images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(24, 5))
    fig.suptitle(f"Predictions vs True Labels")
    for idx in np.arange(len(imgs)):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, xticks=[], yticks=[])

        # Unnormalise image (revive colour)
        for t, m, s in zip(imgs[idx], (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)):
            t.mul_(s).add_(m)

        ax.imshow(imgs[idx].permute(1, 2, 0).numpy())
        ax.set_title(f"{class_labels[preds[idx]]}\n"
                     f"({class_labels[lbls[idx]]})",
                     color=("green" if preds[idx] == lbls[idx] else "red"))
    fig.subplots_adjust(top=0.8, hspace=0.55)
