from pathlib import Path

from app.cleaning import LabelCleaner, FilepathHandler

import pandas as pd
import pytest


@pytest.fixture
def label_cleaner() -> LabelCleaner:
    filenames = {'train_labels': 'train_labels.csv', 'test_labels': 'test_labels.csv'}
    return LabelCleaner(filenames)


@pytest.fixture
def all_label_data(label_cleaner) -> pd.DataFrame:
    return label_cleaner.merge_data()


@pytest.fixture
def labels() -> list[str]:
    return ['civilian', 'military']


def test_valid_merged_data_size(label_cleaner, all_label_data) -> None:
    train_len = len(label_cleaner.labels_train)
    test_len = len(label_cleaner.labels_test)
    assert len(all_label_data) == (train_len + test_len)


def test_valid_update_class_names(label_cleaner, labels, all_label_data) -> None:
    for label in labels:
        all_label_data = label_cleaner.update_class_name(all_label_data, label)

    assert len(all_label_data.groupby('class').count()) == len(labels)


def test_valid_remove_multi_label_data(label_cleaner, all_label_data) -> None:
    updated_data = label_cleaner.remove_multi_label_data(all_label_data)
    diff_len = len(all_label_data) - len(updated_data)

    multi_labelled = all_label_data.groupby('filename').apply(lambda x: len(x['class'].unique())).reset_index()
    multi_labelled = multi_labelled[multi_labelled[0] > 1]
    assert diff_len == len(multi_labelled)


def test_valid_remove_duplicates(label_cleaner, all_label_data) -> None:
    updated_data = label_cleaner.remove_duplicates(all_label_data)
    test_data = label_cleaner.merge_data()
    test_data = test_data.drop_duplicates('filename', keep='last')
    assert len(updated_data) == len(test_data)


def test_valid_create_filepaths(all_label_data) -> None:
    fp_handler = FilepathHandler()
    data_dir = Path(f'{Path.cwd()}/data')

    all_label_data['img_filepaths'] = fp_handler.create_filepaths(all_label_data['filename'])
    test_data = str(data_dir) + '\\' + all_label_data['filename']
    assert test_data.equals(all_label_data['img_filepaths'])

