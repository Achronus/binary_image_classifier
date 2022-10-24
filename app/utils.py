import torch
import torchvision.models as models


class Logger:
    """Stores information generated during model training."""
    def __init__(self) -> None:
        self.keys = ['accuracy', 'valid_loss', 'train_loss']

        for key in self.keys:
            setattr(self, key, [])

    def add(self, names: list[str], values: list[float] | list[int]) -> None:
        """
        Adds a value to the specified data attribute.

        :param names: (list[string]) the list of data attribute to add
        :param values: (list[float] | list[int]) a list of values corresponding to each specified attribute
        """
        for item, value in zip(names, values):
            if item not in self.keys:
                raise ValueError(f"'{item}' does not exist! Valid metrics: '{self.keys}'")

            getattr(self, item).append(value)

    def __repr__(self) -> str:
        return f"Available attributes: '{self.keys}'"


def time_taken(time_diff):
    """
    Calculates the training time taken in hours, minutes and seconds.

    :param time_diff: (float) time difference between start and end
    """
    min_secs, secs = divmod(time_diff, 60)
    hours, mins = divmod(min_secs, 60)
    print(f"Total time taken: {hours:.2f} hrs {mins:.2f} mins {secs:.2f} secs")


def save_model(model: models, filepath: str, logger: Logger) -> None:
    """
    Saves a PyTorch model with utility information.

    :param model: (torchvision.models) model to save
    :param filepath: (string) filepath and name of model
    :param logger: (Logger) a data logger containing model metrics
    """
    torch.save({
        'parameters': model.state_dict(),
        'logger': logger,
        }, filepath)


def load_model(model: models, filepath: str) -> models:
    """
    Loads a PyTorch model with utility information.

    :param model: (torchvision.models) model to load
    :param filepath: (string) filepath and name of saved model
    :returns: the loaded model with a custom attribute 'logger'
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(filepath, map_location=device)
    model.logger = checkpoint['logger']
    model.load_state_dict(checkpoint['parameters'], strict=False)
    return model
