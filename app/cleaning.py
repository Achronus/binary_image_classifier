from pathlib import Path

import pandas as pd


def clean_data() -> pd.DataFrame:
    """Cleans label (if applicable) and image data. Removing duplicate and invalid data, and updates image names
    and moves them to their respective label folder."""
    # Check for labels and data directories
    pass


class FilepathHandler:
    """Handles image filepaths related to the data folder."""
    def __init__(self) -> None:
        self.data_dir = Path(f'{Path.cwd()}/data')
        self.col_filepath = 'img_filepaths'

    def get_filepaths(self, img_extension: str) -> pd.DataFrame:
        """
        Creates a pandas.DataFrame containing the data directory image filepaths based on the given img extension.

        :param img_extension: (string) the file extension for the images in the data folder
        :return: a pandas.DataFrame containing the data directory image filepaths
        """
        data_dir_filepaths = [str(item) for item in list(self.data_dir.glob(f'*.{img_extension}'))]
        return pd.DataFrame({self.col_filepath: data_dir_filepaths})

    def create_filepaths(self, img_filenames: pd.Series) -> pd.Series:
        """
        Combines a pandas.Series of data with the data directory filepath to create new filepaths.

        :param img_filenames: (pandas.Series) a column of data containing image filenames
        :return: a pandas.Series containing new image filepaths, merging the data directory filepath and image filenames
        """
        return str(self.data_dir) + '\\' + img_filenames


class LabelCleaner:
    """
    Cleans existing label data files using Pandas DataFrames.

    :param dataset_filenames: (dict[string, string]) a dictionary containing the train and test label data filenames.
                              These files must be 'csv' files and stored in a 'labels' folder in the root directory.
                              Valid keys - ['train_labels', 'test_labels']
    :param separator: (string, optional) delimiter to use for filenames. Refer to 'pandas.read_csv' 'sep' parameter for
                      accepted separators. Default is ','
    :param filename_column: (string, optional) name of the filename column. Default is 'filename'
    :param label_column: (string, optional) the column name to look for the label. Default is 'class'
    :param custom_columns: (list[string], optional) a list of column names for the cleaned data.
                           Default is ['filename', 'class', 'type']
    """
    def __init__(self, dataset_filenames: dict[str, str], custom_columns: list[str] = None, separator: str = ',',
                 filename_column: str = 'filename', label_column: str = 'class') -> None:
        self.labels_train = pd.read_csv(f'labels/{dataset_filenames["train_labels"]}', sep=separator)
        self.labels_test = pd.read_csv(f'labels/{dataset_filenames["test_labels"]}', sep=separator)
        self.cleaned_data = None  # Updated after calling '.update()'

        self.col_filename = filename_column
        self.col_label = label_column
        self.col_custom = ['filename', 'class', 'type'] if custom_columns is None else custom_columns

    def merge_data(self) -> pd.DataFrame:
        """Combines the train and test label data into a single Pandas DataFrame, adding a column for its type
        (train or test).
        :return: a reduced DataFrame with the train and test label data containing the filename, label, and type columns
        """
        self.labels_train['type'] = 'train'
        self.labels_test['type'] = 'test'
        return pd.concat([self.labels_train, self.labels_test])[self.col_custom].reset_index(drop=True)

    def update(self, labels: list[str], img_extension: str = 'jpg') -> pd.DataFrame:
        """
        Preprocesses the label data, concatenating it together, updating the class names, removing multi-labelled
        instances and missing labelled values.

        :param labels: (list[string]) a list of unique class labels
        :param img_extension: (string) the file extension for the images in the data folder. Default is 'jpg'
        :return: a pandas.DataFrame containing the updated training and test label data
        """
        labels = list(set(labels))
        all_labels = self.merge_data()
        print(f'Total records: {len(all_labels)}')

        # Update class label names
        for label in labels:
            all_labels = self.update_class_name(all_labels, label)

        # Remove invalid and duplicates
        all_labels = self.remove_multi_label_data(all_labels)
        all_labels = self.remove_duplicates(all_labels)
        all_labels = self.remove_data_without_labels(all_labels, img_extension)

        self.cleaned_data = all_labels
        print(f'Total records after cleaning: {len(all_labels)}')
        return all_labels

    def update_class_name(self, data: pd.DataFrame, word: str) -> pd.DataFrame:
        """
        Updates class names in 'data' by removing text before and after the given word.

        :param data: (pandas.DataFrame) a DataFrame containing the training and testing label data
        :param word: (string) the new class label name
        :return: an updated pd.DataFrame
        """
        data[self.col_label] = data[self.col_label].replace({f'.*(?={word})|(?<={word}).*$': ''}, regex=True)
        return data

    def remove_multi_label_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes multi-labelled data instances by grouping by the filename and counting the class instances. Filenames
        with more than one label are then removed from the input data.

        :param data: (pandas.DataFrame) a DataFrame containing the training and testing label data
        :return: an updated pandas.DataFrame without multi-labelled data instances
        """
        data = data.reset_index(drop=True)  # Reset indices first
        label_counts = data.groupby(self.col_filename).apply(lambda x: len(x[self.col_label].unique())).reset_index()
        multi_labelled = label_counts[label_counts[0] > 1]
        print(f'Multi-labelled items removed: {len(multi_labelled)}')
        return data.drop(multi_labelled.index)

    def remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicate data from a given DataFrame of label data.

        :param data: (pandas.DataFrame) a DataFrame containing the training and testing label data
        :return: an updated pandas.DataFrame without duplicate filename instances
        """
        sorted_data = data.sort_values('type', ascending=False)  # train on top, test on bottom
        no_dupes = sorted_data.drop_duplicates(self.col_filename, keep='last').reset_index(drop=True)  # Keep test data
        print(f'Duplicates removed: {len(sorted_data) - len(no_dupes)}')
        return no_dupes

    def remove_data_without_labels(self, data: pd.DataFrame, img_extension: str) -> pd.DataFrame:
        """
        Compares filenames in the given DataFrame with filenames in the data directory, removing records from the
        DataFrame that are not present in both.

        :param data: (pandas.DataFrame) a DataFrame containing the training and testing label data
        :param img_extension: (string) the file extension for the images in the data folder
        :return: an updated pandas.DataFrame without missing label data
        """
        # Store image filepaths from the data folder in a DataFrame
        fp_handler = FilepathHandler()
        col_filepath = fp_handler.col_filepath
        data_dir_filepaths = fp_handler.get_filepaths(img_extension)

        # Create img filepaths in data DataFrame
        updated_data = data.copy()
        updated_data[col_filepath] = fp_handler.create_filepaths(data[self.col_filename])

        # Remove invalid records from DataFrame
        updated_data = data_dir_filepaths.merge(updated_data, how='left', on=[col_filepath]).dropna().reset_index(drop=True)
        print(f'Data without labels removed: {len(data) - len(updated_data)}')
        return updated_data


class ImageCleaner:
    """A class dedicated to modifying image data inside the data folder. Complements the LabelCleaner."""
    pass
