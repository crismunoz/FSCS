from utils import MapAdultDataset, MapCivilCommentsDataset
#from aif360.datasets import AdultDataset
#from transformers import BertTokenizer, logging
from zipfile import ZipFile
import numpy as np
import pandas as pd
import pickle
import os
import shutil
import aif360
import model
import math


# Disable BERT warning about on unused weights. This due to us using BERT
# for a different task than it was originally created for.
logging.set_verbosity_error()

# The continuous columns in the adult dataset.
ADULT_CONTINUOUS = ['age', 'fnlwgt', 'education_num', 'capital_gain',
                    'capital_loss', 'hours_per_week']


def normalization(table, categories):
    """
    The columns of a table are normalized to be between 0 and 1.

    Args:
        table:   The table containing the data that needs to be normalized.
        categories: The list of table columns names to be normalized.
    Returns:
        table:   The table with normalized categories.
    """
    # Loop over the categories to normalize them.
    for column in categories:
        # Find the extreme values of the column.
        min_val = table[column].min()
        max_val = table[column].max()

        # Remap the column values to be between 0 and 1.
        table[column] = (table[column] - min_val) / (max_val - min_val)
    return table


class LoadData(object):
    """
    Gather and process data.
    """
    def __init__(self, data_name) -> None:
        """
        The initialization of the module.

        Args:
            data_name: The name of the dataset that needs to be loaded.
        """
        super(LoadData, self).__init__()

        if data_name == 'adult':

            # Find the aif toolkit path.
            aif_path = os.path.abspath(aif360.__file__)
            aif_path = aif_path.replace('__init__.py', '') + 'data/raw/adult/'

            # Check if train, validation, and test data files are available.
            if not (os.path.isfile(aif_path + 'adult.data') and
                    os.path.isfile(aif_path + 'adult.test') and
                    os.path.isfile(aif_path + 'adult.names')):

                # Find the file directories.
                source = os.path.abspath(os.getcwd()) + "/Data/Adult/"
                get_files = os.listdir(source)

                # Copy the files to the aif toolkit.
                for f in get_files:
                    shutil.move(source + f, aif_path)

            results = self.preprocessAdult()

        elif data_name == 'celeba':
            # Find the path of the correct data.
            path = os.path.abspath(os.getcwd()) + "/Data/CelebA/"

            # Check if the data is still zipped.
            if not (os.path.isdir(path + "img_align_celeba/")):
                zipped = path + "img_align_celeba.zip"

                # Open the zip file in READ mode.
                with ZipFile(zipped, 'r') as zip:
                    # Extract all of the files.
                    print('Extracting all the files for CelebA now.')
                    zip.extractall(path=path)

                print('Done!')

            results = self.preprocessCelebA(path)

        elif data_name == 'civil':
            # Get the path to the data.
            path = os.path.abspath(os.getcwd()) + "/Data/CivilComments/"

            results = self.preprocessCivilComments(path)

        else:
            raise Exception('An invalid dataset name was chosen.')

        self.train = results[0]
        self.val = results[1]
        self.test = results[2]
        self.ratio = results[3]

    def preprocessAdult(self):

        # Load the data from AIF360, as described in the paper.
        ad = AdultDataset(label_name='income-per-year',
                          favorable_classes=['>50K', '>50K.'],
                          protected_attribute_names=['sex'],
                          privileged_classes=[['Male']],
                          categorical_features=['workclass', 'education',
                                                'marital-status', 'occupation',
                                                'relationship',
                                                'native-country', 'race'],
                          features_to_keep=[],
                          features_to_drop=[],
                          na_values=['?'],
                          custom_preprocessing=None,
                          metadata={'label_maps': [
                                        {1.0: '>50K', 0.0: '<=50K'}],
                                    'protected_attribute_maps': [
                                        {1.0: 'Male', 0.0: 'Female'}]})

        # Split the data as stated in the adult.names file.
        train_df_adult, test_df_adult = ad.split([30162])

        # Save the amount of features input.
        model.INPUT_SIZE_ADULT = len(train_df_adult.feature_names) - 1

        # Convert to pandas dataframes.
        train, _ = train_df_adult.convert_to_dataframe()
        test, _ = test_df_adult.convert_to_dataframe()

        # Replace the special chars.
        train.columns = train.columns.str.replace('-', '_')
        test.columns = test.columns.str.replace('-', '_')

        # Drop but the first 50 the female data with high income (d=0 y=1)
        # occurences to introduce bias.
        train.drop(train[(train.sex == 0.0) &
                   (train.income_per_year == 1.0)].index[50:],
                   axis=0, inplace=True)

        # Calculate the protected distribution for the train dataset.
        femaleDistribution = len(train[
                                (train.sex == 0.0)].index) / len(train.index)

        # Normalize the continuous columns in the data.
        train = normalization(train, ADULT_CONTINUOUS)
        test = normalization(test, ADULT_CONTINUOUS)

        # Remove the protected and target data.
        d_train = train.pop('sex')
        y_train = train.pop('income_per_year')
        x_train = train
        d_test = test.pop('sex')
        y_test = test.pop('income_per_year')
        x_test = test

        # Reset the index numbering in pandas.
        d_train.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        x_train.reset_index(drop=True, inplace=True)
        d_test.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)
        x_test.reset_index(drop=True, inplace=True)

        # Load data as a map-style dataset.
        train_data = MapAdultDataset(x_train, y_train, d_train)
        test_data = MapAdultDataset(x_test, y_test, d_test)

        return train_data, None, test_data, femaleDistribution