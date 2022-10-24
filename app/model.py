from dataclasses import dataclass
import numpy as np
import time

from app.utils import Logger, time_taken, save_model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets
import torchvision.models as models
from torch.utils.data import SubsetRandomSampler, DataLoader

from torchmetrics import F1Score, Accuracy, Precision, Recall


@dataclass
class ModelParameters:
    """A data class for storing model parameters."""
    num_classes: int
    batch_size: int
    hidden_layer_sizes: list[int]
    learning_rate: float
    epochs: int
    seed: int


class Classifier(nn.Module):
    """
    A simple classifier with modular layers based on a list of hidden layer sizes.

    :param in_features: (int) the number of input nodes (image pixel size)
    :param out_features: (int) the number of output nodes (classes)
    :param hidden_layers: (list[int]) a list of integers for the size of the hidden layers
    :param drop_prob: (float, optional) dropout rate probability. Default is 0.3
    """
    def __init__(self, in_features: int, out_features: int, hidden_layers: list[int], drop_prob: float = 0.3) -> None:
        super().__init__()
        # Generate a list of layers - adding the first one
        self.hidden_layers = nn.ModuleList([nn.Linear(in_features, hidden_layers[0])])

        # Add remaining hidden layers dynamically
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        # Set the output and dropout layers
        self.out = nn.Linear(hidden_layers[-1], out_features)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagates the features through the network.

        :param x: (torch.Tensor) a tensor of learnable features
        :returns: A torch.Tensor of image probability values for each class label
        """
        # Pass each layer through a ReLU activation and dropout layer
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)

        # Output layer with log softmax applied
        x = F.log_softmax(self.out(x), dim=1)
        return x


class ModelPreparation:
    """
    A class for preparing image data.

    :param model_params: (ModelParameters) a dataclass containing the model parameters
    """
    def __init__(self, model_params: ModelParameters) -> None:
        self.model_params = model_params

        self.train_size = 0  # used in ModelController

        torch.manual_seed(self.model_params.seed)
        np.random.seed(self.model_params.seed)

    def split_data(self, data_filepath: str, train_indices: list[int],
                   test_indices: list[int]) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Applies data augmentations to the dataset from the attribute 'data_filepath' and divides the data into training
        and testing dataloaders based on the predefined attribute indices.

        :param data_filepath: (str) the filepath for the folder containing the image data
        :param train_indices: (list[int]) a list of indices for the training data
        :param test_indices: (list[int]) a list of indices for the test data
        :return: A tuple containing three torch.utils.data.DataLoader for the training, validation, and testing data,
        respectively - (train_loader, valid_loader, test_loader)
        """
        data_transforms = self.__set_transforms()
        dataset = torchvision.datasets.ImageFolder(data_filepath, transform=data_transforms)

        # Set validation data indices
        temp_indices = train_indices.copy()
        split = len(test_indices)
        np.random.shuffle(temp_indices)
        valid_indices = temp_indices[:split]
        train_indices = temp_indices[split:]

        # Set random samplers
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        self.train_size = len(train_sampler)

        # Convert to dataloaders
        train_loader = DataLoader(dataset, batch_size=self.model_params.batch_size, sampler=train_sampler)
        valid_loader = DataLoader(dataset, batch_size=self.model_params.batch_size, sampler=valid_sampler)
        test_loader = DataLoader(dataset, batch_size=self.model_params.batch_size, sampler=test_sampler)

        print(f'Training data: {self.train_size}\n'
              f'Validation data: {len(valid_sampler)}\n'
              f'Test data: {len(test_sampler)}')
        return train_loader, valid_loader, test_loader

    @staticmethod
    def __set_transforms() -> transforms.Compose:
        """Generates a predefined set of image augmentations and returns them as a composition."""
        return transforms.Compose([
            transforms.Resize(224),  # Resize images to 224
            transforms.CenterCrop(224),  # Make images 224x224
            transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip some samples (50% chance)
            transforms.RandomRotation(degrees=20),  # Randomly rotate some samples by 20 degrees
            transforms.ToTensor(),  # Convert image to a tensor
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize image values
        ])


class ModelController:
    """
    A controller class for training and validating models.

    :param model_params: (ModelParameters) a dataclass containing the model parameters
    :param train_loader: (torch.utils.data.DataLoader) a dataloader containing the training data
    :param valid_loader: (torch.utils.data.DataLoader) a dataloader containing the validation data
    """
    def __init__(self, model_params: ModelParameters, train_loader: DataLoader, valid_loader: DataLoader) -> None:
        self.model_params = model_params
        self.model_prep = ModelPreparation(model_params)

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.create_model().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=self.model_params.learning_rate)
        self.criterion = nn.NLLLoss()

        self.logger = Logger()
        self.patience_counter = 0
        self.total_steps = self.model_prep.train_size // self.model_params.batch_size

    def create_model(self) -> models.mobilenet_v3_large:
        """
        Creates a predefined pretrained model, freezes its weights, and attaches a custom classifier to the end.

        :returns: a pretrained large mobilenet_v3 model with a custom classifier
        """
        mobilenetv3 = models.mobilenet_v3_large(weights='DEFAULT')

        # Freeze parameters
        for param in mobilenetv3.parameters():
            param.requires_grad = False

        # Attach classifier
        classifier = Classifier(in_features=mobilenetv3.classifier[0].in_features,
                                out_features=self.model_params.num_classes,
                                hidden_layers=self.model_params.hidden_layer_sizes)
        mobilenetv3.classifier = classifier
        return mobilenetv3

    def train(self, epochs: int, save_filepath: str, iterations: int = 2, patience: int = 10) -> models:
        """
        Trains the predefined model.

        :param epochs: (int) number of training iterations
        :param save_filepath: (string) filepath for saving the model
        :param iterations: (int, optional) number of iterations per epoch. Default is 2
        :param patience: (int, optional) number of epochs to wait before early stopping. Default is 10

        :return: a trained model
        """
        start_time = time.time()
        valid_loss_min = np.Inf
        early_stop = False
        print_every = self.total_steps // iterations

        # Start training loop
        for i_epoch in range(epochs):
            steps, train_loss = 0, 0.

            # Break when reached early stop patience
            if early_stop:
                break

            self.model.train()  # Set model to train mode
            for images, labels in self.train_loader:
                steps += 1
                images, labels = images.to(self.device), labels.to(self.device)

                # Train model
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()  # Update weights (parameters)
                train_loss += loss.item()

                # Output stats
                if steps % print_every == 0:
                    self.model.eval()  # Set model to evaluation mode
                    accuracy, valid_loss = self.validate()

                    # Update and add metrics to logger
                    train_loss = train_loss / print_every
                    valid_loss = valid_loss / len(self.valid_loader)
                    accuracy = float(accuracy / len(self.valid_loader))
                    self.logger.add(
                        names=['accuracy', 'valid_loss', 'train_loss'],
                        values=[accuracy, valid_loss, train_loss]
                    )

                    # Output information
                    print(f"Epoch: {i_epoch + 1}/{epochs}",
                          f"Step: {steps}/{self.total_steps}",
                          f"Training Loss: {train_loss:.3f}",
                          f"Validation Loss: {valid_loss:.3f}",
                          f"Accuracy: {accuracy:.3f}")

                    # Save model with best validation loss
                    if valid_loss <= valid_loss_min:
                        print(f"Validation loss decreased ({valid_loss_min:.3f}",
                              f"-> {valid_loss:.3f}). Saving model...")
                        valid_loss_min = valid_loss
                        save_model(self.model, save_filepath, self.logger)
                        self.patience_counter = 0  # Reset early stop counter

                    # Early stop if patience reached before epochs end
                    elif self.__early_stopping(patience):
                        early_stop = True
                        break

        # Calculate training time
        train_time = (time.time() - start_time)
        time_taken(train_time)
        return self.model

    def validate(self) -> tuple[float, float]:
        """
        Validates the performance of the training model. Performed inside train().

        :return: a tuple containing the accuracy and valid loss as float values - (accuracy, valid_loss)
        """
        with torch.no_grad():
            accuracy, valid_loss = 0., 0.

            for images, labels in self.valid_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # Validate model
                output = self.model.forward(images)
                loss = self.criterion(output, labels)
                valid_loss += loss.item()

                # Calculate accuracy
                probabilities = torch.exp(output)  # log-softmax to probabilities
                score = (labels.data == probabilities.max(dim=1)[1])  # Class with highest probability
                accuracy += torch.Tensor(score).to(torch.float).mean()  # bool -> float

            return accuracy, valid_loss

    def __early_stopping(self, patience: int) -> bool:
        """
        Stops model training early if the validation loss hasn't improved after a given amount of patience
        (epoch iterations).

        :param patience: (int) the number of updates to wait for improvement before termination
        :return: a boolean value to stop training. Can be True or False
        """
        self.patience_counter += 1
        print(f"Early stopping counter: {self.patience_counter}/{patience}.")

        # Stop loop if patience is reached
        if self.patience_counter >= patience:
            print("Early stopping limit reached. Training terminated.")
            return True
        return False


class ModelPredictor:
    """
    A class for making model predictions.

    """
    def __init__(self, model: models, test_data: DataLoader) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.test_data = test_data

    @torch.no_grad()
    def predict(self, seed: int = 136) -> tuple[torch.Tensor, torch.Tensor]:
        """Makes predictions on the test data.

        :param seed: (int) number for setting the random generator, allows reproducibility. Default is 136
        :return: a tuple containing the predictions and image labels, respectively - (y_pred, y_true)
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        y_pred, y_true = torch.Tensor([]), torch.Tensor([])

        for images, labels in self.test_data:
            images, labels = images.to(self.device), labels.to(self.device)

            # Make predictions
            output = self.model.forward(images)
            probabilities = torch.exp(output)
            batch_predictions = probabilities.max(dim=1)[1]

            y_pred = torch.cat((y_pred.to(self.device), batch_predictions), dim=0)
            y_true = torch.cat((y_true.to(self.device), labels), dim=0)
            return y_pred.cpu(), y_true.cpu()

    def compute_stats(self, y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int = 2) -> dict[str, float]:
        """
        Computes a set of statistics for the model. Dictionary keys - ['precision', 'recall', 'accuracy', 'f1_score'].

        :param y_pred: (torch.Tensor) model predictions
        :param y_true: (torch.Tensor) image labels
        :param num_classes: (int, optional) the number of classes. Default is 2
        :return: a dictionary containing classification statistics
        """
        y_pred, y_true = y_pred.to(torch.int), y_true.to(torch.int)

        precision = Precision(num_classes=num_classes)(y_pred, y_true)
        recall = Recall(num_classes=num_classes)(y_pred, y_true)
        accuracy = Accuracy(num_classes=num_classes)(y_pred, y_true)
        f1_score = F1Score(num_classes=num_classes)(y_pred, y_true)

        return {
            'precision': precision.item(),
            'recall': recall.item(),
            'accuracy': accuracy.item(),
            'f1_score': f1_score.item()
        }
