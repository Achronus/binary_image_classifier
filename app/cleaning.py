import pandas as pd


class LabelCleaner:
    """
    A class dedicated to cleaning training and testing labels Pandas DataFrames.

    :param train_labels: (pd.DataFrame) a pandas DataFrame containing the training label data
    :param test_labels: (pd.DataFrame) a pandas DataFrame containing the test label data
    :param labels: (list[str]) a list of unique class names. E.g., ['civilian', 'military']
    :param filename_column: (string, optional) name of the filename column. Default is 'filename'
    :param label_column: (string, optional) the column name to look for the label. Default is 'class'
    """

    def __init__(self, train_labels: pd.DataFrame, test_labels: pd.DataFrame, labels: list[str],
                 filename_column: str = 'filename', label_column: str = 'class') -> None:
        self.labels = list(set(labels))
        self.label_column = label_column

        self.single_data_cleaner = SingleDataCleaner(
            train_labels,
            test_labels,
            labels=labels,
            filename_column=filename_column,
            label_column=label_column
        )

    def clean_data(self) -> pd.DataFrame:
        """
        Preprocesses the class data using various helper functions.

        :return: a pd.DataFrame containing the updated training and test label data
        """
        self.single_data_cleaner.remove_individual_duplicates()
        data_labels = self.single_data_cleaner.merge_labels().reset_index(drop=True)

        for label in self.labels:
            data_labels = self.__update_class_name(data_labels, label)

        return data_labels

    def __update_class_name(self, data: pd.DataFrame, word: str) -> pd.DataFrame:
        """
        Updates class names in 'data' by removing text before and after the given word.

        :param word: (string) the new class label name
        :return: an updated pd.DataFrame
        """
        data[self.label_column] = data[self.label_column].replace({
            f'.*(?={word})|(?<={word}).*$': ''
        }, regex=True)
        return data


class SingleDataCleaner:
    """A helper class for performing initial cleaning of the training and test labels separately.

    :param train_labels: (pd.DataFrame) a pandas DataFrame containing the training label data
    :param test_labels: (pd.DataFrame) a pandas DataFrame containing the test label data
    :param labels: (list[str]) a list of unique class names. E.g., ['civilian', 'military']
    :param filename_column: (string) name of the filename column.
    :param label_column: (string) the column name to look for the label.
    """
    def __init__(self, train_labels: pd.DataFrame, test_labels: pd.DataFrame, labels: list[str],
                 filename_column: str, label_column: str) -> None:
        self.train_labels = train_labels
        self.test_labels = test_labels

        self.labels = list(set(labels))
        self.filename_column = filename_column
        self.label_column = label_column

    def get_label_data(self, data: pd.DataFrame, label: str) -> pd.DataFrame:
        """
        Retrieves data from the class dataframe based on the given label and column name.

        :param data: (pd.DataFrame) the training or testing labels DataFrame
        :param label: (string) a single class label found in the data
        :return: a new pd.DataFrame containing all rows and columns for the desired class label
        """
        return data[data[self.label_column].str.contains(label)]

    def merge_labels(self) -> pd.DataFrame:
        """
        Merges the train and test label datasets into a single DataFrame.

        :returns: a pd.DataFrame containing a combination of the train and test labels
        """
        return pd.concat([self.train_labels, self.test_labels]).drop_duplicates(self.filename_column, keep='last')

    def remove_individual_duplicates(self) -> None:
        """
        Removes overlapping class names (images with both classes) and duplicate filenames from each dataset,
        respectively. Also, adds a 'type' column, specifying whether the data is for training or testing.
        Updates the class attributes - self.train_labels, self.test_labels.
        """
        for key, dataset in {'train_labels': self.train_labels, 'test_labels': self.test_labels}.items():
            filenames_label_a = self.get_label_data(dataset, self.labels[0])
            filenames_label_b = self.get_label_data(dataset, self.labels[1])

            # Clean data (overlapping class names and duplicate filename)
            overlap_duplicates = filenames_label_b[
                filenames_label_b[self.filename_column].isin(filenames_label_a[self.filename_column])
            ].drop_duplicates(self.filename_column)
            filenames_unique = dataset.drop_duplicates(self.filename_column)
            filenames_clean = filenames_unique[
                ~filenames_unique[self.filename_column].isin(overlap_duplicates[self.filename_column])
            ].reset_index(drop=True)

            # Add data type column
            temp = filenames_clean.copy()
            temp['type'] = key.split('_')[0]

            print(f'{key} overlap duplicates removed: {len(overlap_duplicates)}')
            setattr(self, key, temp)
