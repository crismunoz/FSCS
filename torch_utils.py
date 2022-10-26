import torch
import torch.nn as nn
import random

NUM_GROUPS = 2

class FeaturizerPhi(nn.Module):
    """
    The featurizer Phi module, which extracts the features from the data.
    """
    def __init__(self, layers) -> None:
        """
        The initialization of the module.

        Args:
            layers: The layers needed for this module depending on the dataset.
        """
        super(FeaturizerPhi, self).__init__()
        self.layers = layers

    def forward(self, x):
        """
        The forward step of the module.

        Args:
            x:  The data for which to extract the features.
        Returns:
            out: The extracted features of the data.
        """
        return self.layers(x)


class TwoLayerNN(nn.Module):
    """
    Two layer network that will form the sufficiency regularized model.
    Contains the joint classifier, along with the group- and random splits.
    """
    def __init__(self, hidden_size, phi, nb_classes, ratio, pretrained=None) -> None:
        """
        The initialization of the two layer network.

        Args:
            name:           The name of the dataset that needs to be trained.
            nb_classes:    The amount of target classes in the dataset.
            ratio:          Ratio of protected group datapoints within dataset.
            model_size:     Size of BERT model to use on CivilComments dataset.
        """
        super(TwoLayerNN, self).__init__()
        self.nb_classes = nb_classes
        self.ratio = ratio

        # Obtain hidden layer size, a pretrained network, if any,
        # and the featurizer.
        self.hidden_size = hidden_size
        self.pretrained = pretrained
        self.phi = phi 

        self.groups = nn.ModuleList([
                        nn.Linear(hidden_size, nb_classes) for _ in range(NUM_GROUPS)])
        self.joint = nn.Linear(hidden_size, nb_classes)

    def forward_group(self, X, D, X_mask=None):
        """
        The forward for the fully connected group layers.

        Args:
            X:      The input data to the module.
            D:      The group of the input data.
            X_mask: Attention mask used for BERT model only.
        Returns:
            group_preds: Group-specific predictions to be used
                         for group-specific aggregate loss.
        """
        # Check if there is a BERT module before the featurizer.
        if self.pretrained:
            X = self.pretrained(X, X_mask)[1]

        # Receive features from featurizer.
        phi = self.phi(X)#.squeeze()

        # Prepare a tensor of correct shape for predictions.
        group_preds = torch.zeros(phi.shape[0],
                                  self.nb_classes, device=self.device)

        # Make predictions using the correct layer depending on group d.
        for i, (x, d) in enumerate(zip(phi, D)):
            group_preds[i] = self.groups[int(d.item())](x)

        return group_preds

    def forward(self, X, D, X_mask=None):
        """
        The forward pass of the module. Here X is transformed
        through several layer transformations.

        Args:
            X:      The input data to the module.
            D:      The protected attribute of the input data.
            X_mask: Attention mask used for BERT model only.
        Returns:
            jointPreds:             The output predictions by the
                                    sufficiency regularized joint classifier.
            group_specific_preds:   The group specific input predictions.
            group_agnostic_preds:   The group agnostic input predictions.
        """
        # Check if there is a network before the featurizer.
        if self.pretrained:
            X = self.pretrained(X, X_mask)[1]

        # Receive features from featurizer.
        phi = self.phi(X)#.squeeze()

        # Prepare tensors of correct shapes for predictions.
        group_specific_preds = torch.zeros(phi.shape[0], self.nb_classes,
                                           device=self.device)
        group_agnostic_preds = torch.zeros(phi.shape[0], self.nb_classes,
                                           device=self.device)

        # Make group specific and group agnostic predictions with the
        # fully connected layers. For which the former depends on the
        # protected attribute of the data and the latter on random
        # choise given the ratio of the protected and unprotected attribute.
        for i, (x, d) in enumerate(zip(phi, D)):
            group_specific_preds[i] = self.groups[int(d.item())](x)
            group_agnostic_preds[i] = self.groups[1
                                                  if random.random() < self.ratio
                                                  else 0](x)

        # Classify the features found by the featurizer.
        jointPreds = self.joint(phi)

        return jointPreds, group_specific_preds, group_agnostic_preds

    @property
    def device(self):
        """
        Returns the device on which the model is.
        """
        return next(self.parameters()).device


class Baseline(nn.Module):
    """
    The baseline module for the datasets. Not sufficiency regularized.
    """
    def __init__(self, hidden_size, pretrained, phi, num_classes) -> None:
        """
        The initialization of the module.

        Args:
            name:           The name of the dataset.
            num_classes:    The amount of classes the dataset has.
            X_mask:         Attention mask used for BERT model only.
        """
        super(Baseline, self).__init__()

        # Get the featurizer, hidden_size and pretrained module.
        self.pretrained = pretrained
        self.phi = phi 

        # Define the classifier layer.
        self.joint = nn.Linear(hidden_size, num_classes)

    def forward(self, X, X_mask=None):
        """
        Performs forward pass of the input. Here an input tensor x is
        transformed through several layer transformations.

        Args:
            X:      The data input to the module.
            X_mask: Attention mask used for BERT model only.
        Returns:
            pred:   The output predictions of the module.
        """
        # Check if there is a network before the featurizer.
        if self.pretrained is not None:
            X = self.pretrained(X, X_mask)[1]

        # Receive features from featurizer.
        phi = self.phi(X).squeeze()

        # Pass the features through the classifier.
        return self.joint(phi)

    @property
    def device(self):
        """
        Returns the device on which the model is.
        """
        return next(self.parameters()).device