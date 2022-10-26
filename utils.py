import torch


class MapAdultDataset(object):
    """
    The map-style dataset object for the Adult dataset, which is used
    as input for the pytorch Dataloader constructor.
    """
    def __init__(self, x, y, d) -> None:
        """
        Initialization of the dataset object.

        Args:
            x:  The dataframe containing the data for the Adult dataset.
            y:  The target values dataframe.
            d:  The protected attribute dataframe.
        """

        # Turn the provided dataframes into tensors.
        self.x = torch.tensor(x.values, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.d = torch.tensor(d, dtype=torch.long)

    def __len__(self):
        """
        Function that detemines the length of the object.

        Returns:
            lenght: The lenght of the object.
        """
        return len(self.y)

    def __getitem__(self, idx):
        """
        Function that gathers the requested data and returns it.

        Args:
            idx:    The index of the object that is requested.
        Returns:
            x:  The data that corresponds to the index.
            y:  The target value of the requested index.
            d:  The protected attribute that corresponds to the index.
        """
        return self.x[idx], self.y[idx], self.d[idx]


class MapCivilCommentsDataset(object):
    """
    The map-style dataset object for the Civil dataset, which is used
    as input for the pytorch Dataloader constructor.
    """
    def __init__(self, x, y, d) -> None:
        """
        Initialization of the dataset object.

        Args:
            x:  Tensor of the data for the Civil dataset.
            y:  The target values dataframe.
            d:  The protected attribute dataframe.
        """
        self.x = x
        self.y = torch.tensor(y, dtype=torch.float32)
        self.d = torch.tensor(d, dtype=torch.long)

    def __len__(self):
        """
        Function that detemines the length of the object.

        Returns:
            lenght: The lenght of the object.
        """
        return len(self.y)

    def __getitem__(self, idx):
        """
        Function that gathers the requested data and returns it.

        Args:
            idx:    The index of the object that is requested.
        Returns:
            x:  The data that corresponds to the index.
            y:  The target value of the requested index.
            d:  The protected attribute that corresponds to the index.
        """
        return self.x[idx], self.y[idx], self.d[idx]
