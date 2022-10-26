from tqdm import tqdm
from torch_utils import Baseline
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class TrainArgs:
    """
    training configuration for Trainer Class
    """

    def __init__(
        self,
        initial_epoch=0,
        epochs=10,
        adversary_loss_weight=0.1,
        initial_lr=0.001,
        verbose=0,
        print_interval=1,
        batch_size=32,
        shuffle_data=True,
    ):
        self.epochs = epochs
        self.adversary_loss_weight = adversary_loss_weight
        self.initial_lr = initial_lr
        self.initial_epoch = initial_epoch
        self.verbose = verbose
        self.print_interval = print_interval
        self.batch_size = batch_size
        self.shuffle_data = shuffle_data
        
class Trainer:
    def __init__(self, train_args, model, dataset):
        self.train_args = train_args
        self.model = model 
        self.train_loader = self._build_dataloader(dataset)
    
    def _build_dataloader(self, dataset):
        """Build loss function and optimizers"""

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.train_args.batch_size,
            shuffle=self.train_args.shuffle_data,
        )
        return dataloader
        
    def train(self):
        # Set the default device.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model and loss module.
        loss_module = nn.CrossEntropyLoss()

        # Set the model on the correct device.
        self.model.to(device)

        # Set the model in training mode.
        self.model.train()

        # Define a simple Adam optimizer.
        optimizer = optim.Adam(self.model.parameters(), lr=self.train_args.initial_lr)

        # Initialize a variable.
        best_acc = 0

        # Loop over epochs using tqdm to show progress.
        for e in tqdm(range(self.train_args.epochs), leave=True, disable=self.train_args.verbose):

            # Loop over the batches in the Dataloader.
            for inputs, y, d in self.train_loader:
                # If there is a BERT model before the network,
                # find the mask and data appropriately.
                inputs = {val_name:val.to(device) for val_name,val in inputs.items()}
                
                # Move the data to device.
                y = y.to(device)
                d = d.to(device)

                # Perform a forward pass on the data with the model.
                joint_preds = self.model(**inputs)

                # Calculate the joint loss of the predictions.
                l0 = loss_module(joint_preds, y.long().flatten())

                # Before calculating the gradients, we need to ensure that they
                # are all zero. The gradients would not be overwritten, but
                # actually added to the previous ones otherwise.
                optimizer.zero_grad()

                # Add the joint backward to the optimizer.
                l0.backward()

                # Take an optimization step of the parameters.
                optimizer.step()
    
    
class TrainerFSCS:
    def __init__(self, train_args, model, dataset, lambd=0.7):
        self.train_args = train_args
        self.model = model 
        self.train_loader = self._build_dataloader(dataset)
        self.lambd = lambd
    
    def _build_dataloader(self, dataset):
        """Build loss function and optimizers"""

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.train_args.batch_size,
            shuffle=self.train_args.shuffle_data,
        )
        return dataloader

    def train(self):
        # Set the default device.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model and loss module.
        loss_module = nn.CrossEntropyLoss()

        # Set the model on the correct device.
        self.model.to(device)

        # Set the model in training mode.
        self.model.train()

        # Load the parameters for all layers.
        phi_parameters = self.model.phi.parameters()
        group_parameters = [parameters for group in self.model.groups
                            for parameters in group.parameters()]
        joint_parameters = self.model.joint.parameters()

        # Define a simple Adam optimizer for each set of parameters.
        phi_optimizer = optim.Adam(phi_parameters, lr=self.train_args.initial_lr, weight_decay=0)
        group_optimizer = optim.Adam(group_parameters, lr=self.train_args.initial_lr, weight_decay=0)
        joint_optimizer = optim.Adam(joint_parameters, lr=self.train_args.initial_lr, weight_decay=0)

        # Loop over epochs using tqdm to show progress.
        for e in tqdm(range(self.train_args.epochs), leave=True, disable=self.train_args.verbose):

            # The group specific loop over the data.
            for inputs, y, d in self.train_loader:

                # If there is a BERT model before the network,
                # find the mask and data appropriately.
                inputs = {val_name:val.to(device) for val_name,val in inputs.items()}

                # Move the data to device.
                y = y.to(device)
                d = d.to(device)

                # Before calculating the gradients, we need to ensure that they
                # are all zero. The gradients would not be overwritten, but
                # actually added to the previous ones otherwise.
                group_optimizer.zero_grad()

                # Perform the forward of the group layers and calculate the loss.
                group_specific_preds = self.model.forward_group(**inputs, D=d)
                group_specific_loss = loss_module(group_specific_preds.float(),
                                                y.long().flatten())

                # Perform the backward of the loss and optimizer step.
                group_specific_loss.backward()
                group_optimizer.step()

            # The featurizer and joint classifier loop over the data.
            for inputs, y, d in self.train_loader:

                # If there is a BERT model before the network,
                # find the mask and data appropriately.
                inputs = {val_name:val.to(device) for val_name,val in inputs.items()}

                # Move the data to device.
                y = y.to(device)
                d = d.to(device)

                # Perform the model forward and return the predictions.
                joint_preds, group_specific_preds, group_agnostic_preds = self.model(**inputs, D=d)

                # Calculate the joint loss l0 and regularizer loss lr.
                l0 = loss_module(joint_preds, y.long().flatten())
                lr = self.lambd * (loss_module(group_specific_preds,  y.long().flatten()) -
                            loss_module(group_agnostic_preds,  y.long().flatten()))

                # Before calculating the gradients, we need to ensure that they
                # are all zero. The gradients would not be overwritten, but
                # actually added to the previous ones otherwise.
                phi_optimizer.zero_grad()

                # Already perform the backward propegation for the regularizer loss.
                lr.backward(retain_graph=True)

                # Set the joint gradients to zero.
                joint_optimizer.zero_grad()

                # Add the joint backward to both optimizers.
                l0.backward()

                # Take a step with both optimizers.
                joint_optimizer.step()
                phi_optimizer.step()
                
            
def train_base(num_classes, lr, epochs, train_loader, val_loader, progress_bar, model_size):
    """
    Training function that trains the Baseline model with the given data.

    Args:
        num_classes:    The number of classes to predict.
        lr:     Learning rate used for the Adam optimizer.
        epochs: Number of training epochs to perform.
        train_loader:   The training pytorch Dataloader of the dataset.
        val_loader:     The validation pytorch Dataloader.
        progress_bar:   The progress_bar flag for tqdm.
        model_size:     The size for the BERT pretrained model used.
    Returns:
        model:  The trained Baseline model.
    """

    # Set the default device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model and loss module.
    model = Baseline(num_classes, model_size)
    loss_module = nn.CrossEntropyLoss()

    # Set the model on the correct device.
    model.to(device)

    # Set the model in training mode.
    model.train()

    # Define a simple Adam optimizer.
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize a variable.
    best_acc = 0

    # Loop over epochs using tqdm to show progress.
    for e in tqdm(range(epochs), leave=True, disable=progress_bar):

        # Loop over the batches in the Dataloader.
        for x, y, d in train_loader:

            # If there is a BERT model before the network,
            # find the mask and data appropriately.
            if name == 'civil':
                x_mask = x['attention_mask'].to(device)
                x = x['input_ids'].squeeze(1).to(device)
            else:
                x_mask = None
                x = x.to(device)

            # Move the data to device.
            y = y.to(device)
            d = d.to(device)

            # Perform a forward pass on the data with the model.
            joint_preds = model(x, x_mask)

            # Calculate the joint loss of the predictions.
            l0 = loss_module(joint_preds, y.long())

            # Before calculating the gradients, we need to ensure that they
            # are all zero. The gradients would not be overwritten, but
            # actually added to the previous ones otherwise.
            optimizer.zero_grad()

            # Add the joint backward to the optimizer.
            l0.backward()

            # Take an optimization step of the parameters.
            optimizer.step()

        # Check for validation data.
        if val_loader is not None:
            model, accuracy, _, _ = test_base(name, model, val_loader, 0.0)
            print("acc_val:", accuracy)

            # Save the model with the best validation accuracy as it
            # generalizes best.
            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(model.state_dict(), './Models/{}_base.pt'.format(name))

            # Put the model in training mode.
            model.train()
        else:
            torch.save(model.state_dict(), './Models/{}_base.pt'.format(name))

    # Load the best model parameters into the model before returning it.
    model.load_state_dict(torch.load('./Models/{}_base.pt'.format(name)))

    return model