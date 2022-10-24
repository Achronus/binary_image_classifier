from pathlib import Path

import pandas as pd


class ImageHandler:
    """
    A class dedicated to updating image data and separating them into folders based on their class label.

    :param data_labels: (pd.DataFrame) a pandas DataFrame containing the training and testing label data with columns -
                         ['filename', 'old_filename', 'class', 'type']
    :param class_labels: (list[str]) a list of label class names. E.g., ['civilian', 'military']
    :param filename_column: (string, optional) name of the filename column. Set to 'filename' by default
    :param label_column: (string, optional) the column name to look for the label. Set to 'class' by default
    """

    def __init__(self, data_labels: pd.DataFrame, class_labels: list[str], filename_column: str = 'filename',
                 label_column: str = 'class') -> None:
        self.data_labels = data_labels
        self.labels = list(set(class_labels))

        self.filename_column = filename_column
        self.label_column = label_column
        self.data_dir = Path(f'{Path.cwd()}/data')
        self.filepath_column = 'img_filepaths'
        self.updated_labels_filepath = 'labels/updated_labels.csv'

        self.__create_label_dirs()
        self.__add_filename_paths()  # Update 'self.data_labels' with new column

    def __create_label_dirs(self) -> None:
        """Creates label directories in the data folder if they do not already exist."""
        for label in self.labels:
            Path(f'{self.data_dir}/{label}').mkdir(exist_ok=True)

    def update_data_dir(self) -> None:
        """A controller that updates the data folder and 'data_labels' by removing invalid images and
        changing filenames."""
        self.__remove_images()
        self.__update_filename()

        filenames = self.data_labels[self.filename_column].to_list()

        self.__update_img_names(filenames)
        self.__move_imgs(filenames)

    def __update_img_names(self, filenames: list[str]) -> None:
        """Updates image filenames in the 'data' folder."""
        data_folder_img_paths = self.get_data_folder_img_names()
        filepaths = self.data_labels[self.filepath_column].to_list()

        if len(data_folder_img_paths) > 0:
            # Iterate over filenames and filepaths
            for name, old_path in zip(filenames, filepaths):
                new_img_name = f'{self.data_dir}\\{name}'
                Path(old_path).replace(new_img_name)  # Update image names

    def __move_imgs(self, filenames: list[str]) -> None:
        """Moves images in the 'data' folder into their respective class label folders."""
        data_folder_img_paths = self.get_data_folder_img_names()  # Get updated img paths
        filepaths = data_folder_img_paths[self.filepath_column]

        if len(data_folder_img_paths) > 0:
            # Get length of label data
            len_label_a = len(self.data_labels[filepaths.str.contains(self.labels[0])])
            len_label_b = len(self.data_labels[filepaths.str.contains(self.labels[1])])

            # Move images to respective label directories
            for name, old_path in zip(filenames, filepaths.to_list()):
                label_dir = self.labels[0] if self.labels[0] in old_path else self.labels[1]
                img_to_label_dir_path = f'{self.data_dir}\\{label_dir}\\{name}'
                Path(old_path).replace(img_to_label_dir_path)

            print(f'Moved: {len_label_a} imgs -> {self.labels[0]} directory, '
                  f'{len_label_b} imgs -> {self.labels[1]} directory')
        else:
            print('Data directory has been organised! No updates required.')

    def __update_filename(self) -> None:
        """
        Updates the filenames within 'data_labels', converting it into an index with its respective class label
        name. Also, creates a new column with the new filepaths.
        """
        split_data = []
        for label in self.labels:
            # Get label data
            label_data = self.data_labels[self.data_labels[self.label_column].str.contains(label)].reset_index(
                drop=True)

            # Convert to new name '[class]_[class_idx].jpg'
            label_data[self.filename_column] = label_data[self.label_column] + '_' + \
                                               label_data[self.label_column].index.astype(str) + '.jpg'
            split_data.append(label_data)

        # Merge updated label data
        self.data_labels = pd.concat(split_data).reset_index(drop=True)

    def __add_filename_paths(self) -> None:
        """
        Adds two columns for the 'filepath_column' attributes to 'data_labels' with the filename paths.
        """
        str_dir = str(self.data_dir) + '\\'
        temp_labels = self.data_labels.copy()
        temp_labels[self.filepath_column] = str_dir + temp_labels[self.filename_column]
        self.data_labels = temp_labels

    def get_data_folder_img_names(self) -> pd.DataFrame:
        """Retrieves the image names in the 'data' folder and returns them as a pd.DataFrame."""
        img_filepaths = [str(item) for item in list(self.data_dir.glob('*.jpg'))]
        return pd.DataFrame({self.filepath_column: img_filepaths})

    def __remove_images(self) -> None:
        """Deletes invalid images."""
        img_filepaths = self.get_data_folder_img_names()
        invalid = pd.concat([img_filepaths, self.data_labels]).drop_duplicates([self.filepath_column], keep=False)
        total_with_duplicates = len(img_filepaths) + len(self.data_labels)

        # Temp bug fix - Manually remove three troublesome records
        temp_records = ['1convoy.jpg', 'h1.jpg', 'ripsaw.jpg']
        for record in temp_records:
            self.data_labels = self.data_labels[~self.data_labels[self.filepath_column].str.contains(record)]

        # If invalid is larger than 0 and not equal to duplicate total
        if 0 < len(invalid) != total_with_duplicates:
            print(f'Total invalid: {len(invalid)}')
            # Update data labels
            duplicate_indices = self.data_labels[
                self.data_labels[self.filepath_column].isin(invalid[self.filepath_column])
            ].index.to_list()
            self.data_labels = self.data_labels.drop(index=duplicate_indices).reset_index(drop=True)

            # Remove images
            for item in invalid[self.filepath_column].to_list():
                Path(item).unlink(missing_ok=True)

            # Store data labels
            self.data_labels.to_csv(self.updated_labels_filepath)
