import copy
import random


import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from models import get_model
from utils import log_info, set_random_seed, evaluate


class FedACGClient:
    """
    Federated Averaging client.
    """

    def __init__(self, client_id, data_loader, model_parameter, args):
        """
        Initialize a FedACGClient instance.

        Parameters:
            client_id: Identifier for the client.
            data_loader: DataLoader for the client's data.
            model_parameter: Initial global model parameters.
            args: Configuration arguments.
        """
        self.args = args
        set_random_seed(self.args.seed * (self.args.round + 1))
        self.client_id = client_id
        self.data_loader = data_loader
        self.model = get_model(args).to(args.device)
        self.global_model_param = model_parameter

        self.model.load_state_dict(self.global_model_param)
        self.criterion = nn.CrossEntropyLoss().to(args.device)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.args.lr * (self.args.lr_decay**self.args.round),
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        self.beta = self.args.fedacg_beta

    def train(self):
        """
        Train the client model locally.

        Returns:
            Average training loss.
        """
        total_number, labels = self.get_data_stats()
        log_info(
            f"\tStarting local training client {str(self.client_id).zfill(3)} "
            f"(# of data: {total_number} / labels: {labels})"
        )

        self.model.train()
        total_loss = 0
        step_count = 0

        for epoch in range(self.args.epochs):
            for inputs, labels in self.data_loader:
                inputs, labels = inputs.to(self.args.device), labels.to(
                    self.args.device
                )
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                step_count += 1
                # Compute prox term: ||theta_local - theta_global_shifted||^2 over all parameters.
                prox_loss = 0.0
                for name, param in self.model.named_parameters():
                    prox_loss += torch.sum((param - self.global_model_param[name]) ** 2)

                loss = loss + self.beta * prox_loss

                loss.backward()
                self.optimizer.step()

        self.local_evaluate()
        return total_loss / step_count if step_count else 0

    def local_evaluate(self):
        """
        Evaluate the client model.

        Returns:
            Accuracy percentage.
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.data_loader:
                inputs, labels = inputs.to(self.args.device), labels.to(
                    self.args.device
                )
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total if total > 0 else 0
        log_info(
            f"\tLocal evaluation accuracy for client {str(self.client_id).zfill(3)}: {accuracy:.2f}%"
        )
        return accuracy

    def get_gradients(self):
        """
        Compute gradients based on the difference between local and global model parameters.

        Returns:
            Dictionary of gradients.
        """
        gradients = {}
        for name, param in self.model.state_dict().items():
            gradients[name] = param - self.global_model_param[name]
        return gradients

    def set_model(self, parameters):
        """
        Set model weights.

        Parameters:
            parameters: State dictionary to load into the model.
        """
        self.model.load_state_dict(parameters)

    def get_data_stats(self):
        """
        Calculate data statistics for the client.

        Returns:
            Tuple of (total number of data points, sorted list of unique labels).
        """
        total_number = sum(len(data[1]) for data in self.data_loader)
        labels = sorted(
            {label.item() for data in self.data_loader for label in data[1]}
        )
        return total_number, labels


class FedACG:
    """
    Federated Averaging algorithm.
    """

    def __init__(self, args, train_data, test_data):
        """
        Initialize FedACG.

        Parameters:
            args: Configuration arguments.
            train_data: List or dictionary of client training data.
            test_data: Global test data.
        """
        self.args = args
        self.train_data = train_data
        self.test_data = test_data
        self.global_model = get_model(args).to(args.device)
        self.client_ids = list(range(args.num_clients))
        self.global_lr = 1.0
        self.round = 0
        self.args.server_momentum = self.args.fedacg_server_momentum
        # Initialize global momentum and delta (all zeros with same shape as model parameters)
        self.global_momentum = {
            layer: torch.zeros_like(param)
            for layer, param in self.global_model.state_dict().items()
        }
        self.global_delta = {
            layer: torch.zeros_like(param)
            for layer, param in self.global_model.state_dict().items()
        }

    def select_clients(self):
        """
        Randomly select a subset of clients for training.

        Returns:
            Sorted list of selected client IDs.
        """
        random.seed(self.args.seed * (self.round + 1))
        num_clients_to_select = self.args.selected_clients
        return sorted(random.sample(self.client_ids, num_clients_to_select))

    def aggregate_gradients(self, client_updates):
        """
        Aggregate client gradients and update global model.

        Parameters:
            client_updates: List of client update dictionaries.
        """
        aggregated_weight = copy.deepcopy(self.global_model.state_dict())
        global_model_delta = {}
        for layer in aggregated_weight.keys():
            layer_gradients = torch.stack(
                [
                    client_update["update"][layer].float()
                    for client_update in client_updates
                ]
            )
            mean_layer_gradients = torch.mean(layer_gradients, dim=0)
            aggregated_weight[layer] = (
                aggregated_weight[layer] + self.global_lr * mean_layer_gradients
            )
            global_model_delta[layer] = self.global_lr * mean_layer_gradients

        # Update momentum for each layer.
        for layer in aggregated_weight.keys():
            delta = global_model_delta[layer]
            self.global_momentum[layer] = (
                self.args.server_momentum * self.global_momentum[layer] + delta
            )

        self.global_model.load_state_dict(aggregated_weight)

    def train(self):
        """
        Train the global model over multiple rounds.
        """
        for current_round in range(self.args.rounds):
            self.round = current_round
            self.args.round = current_round
            log_info(f"R:{str(current_round).zfill(4)} Starting global round")
            selected_client_ids = self.select_clients()
            # --- FedACG Lookahead: shift the global model before sending it ---
            shifted_model = copy.deepcopy(self.global_model.state_dict())
            for layer in shifted_model.keys():
                shifted_model[layer] = (
                    shifted_model[layer]
                    + self.args.server_momentum * self.global_momentum[layer]
                )

            arguments = [
                (
                    client_id,
                    copy.deepcopy(shifted_model),
                    self.train_data[client_id],
                    self.args,
                )
                for client_id in selected_client_ids
            ]

            client_updates = [self.client_train(arg) for arg in arguments]

            log_info(f"R:{str(current_round).zfill(4)} Global aggregation")
            total_loss = sum(update["loss"] for update in client_updates) / len(
                client_updates
            )

            wandb.log({"training_loss": total_loss}, step=self.round)

            if self.args.send_gradients:
                self.aggregate_gradients(client_updates)
            else:
                self.aggregate_weights(client_updates)

            evaluate(self.global_model, self.test_data, current_round, self.args)

    def client_train(self, params):
        """
        Train a client and return its update.

        Parameters:
            params: Tuple containing (client_id, model_parameter, data_loader, args).

        Returns:
            Dictionary with client_id, update, and loss.
        """
        client_id, model_parameter, data_loader, args = params
        client = FedACGClient(client_id, data_loader, model_parameter, args)
        loss = client.train()

        update = client.get_gradients()

        return {
            "client_id": client_id,
            "update": update,
            "loss": loss,
        }
