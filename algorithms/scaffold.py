import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import pickle
from models import get_model
from utils import log_info, set_random_seed, evaluate
import os
import shutil


class SCAFFOLDClient:
    """
    Federated Averaging client.
    """

    def __init__(
        self, client_id, data_loader, model_parameter, global_control_variate, args
    ):
        """
        Initialize a SCAFFOLDClient instance.

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

        # SCAFFOLD specific implementation
        self.local_state_file = (
            f"data/local_state/{wandb.run.id}/client_{self.client_id}_state.pkl"
        )
        self.global_control_variate = global_control_variate
        self.local_control_variate = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
        }
        self.local_control_variate_plus = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
        }
        self.local_step = 0

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

                loss.backward()
                # Correct the gradient using control variates
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        param.grad.data = (
                            param.grad.data
                            - self.local_control_variate[name]
                            + self.global_control_variate[name]
                        )
                self.optimizer.step()

        self.local_step = step_count
        # self.local_evaluate()
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

    def get_update(self):
        # https://github.com/Lee-Gihun/FedNTD/blob/be00ee598139654abe650c446a7ba3bc4e340233/algorithms/scaffold/ClientTrainer.py#L94
        w_d = {}
        c_d = {}
        rho = 0.9
        self.local_step = (
            self.local_step - rho * (1.0 - pow(rho, self.local_step)) / (1.0 - rho)
        ) / (1.0 - rho)
        for name, param in self.model.state_dict().items():
            w_d[name] = param - self.global_model_param[name]
            lr = self.args.lr
            self.local_control_variate_plus[name] = (
                self.local_control_variate[name]
                - self.global_control_variate[name]
                + (self.global_model_param[name] - param) / (lr * self.local_step)
            )
            c_d[name] = (
                self.local_control_variate_plus[name] - self.local_control_variate[name]
            )
        self.save_local_state()
        return w_d, c_d

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

    def save_local_state(self):

        local_state = {
            "local_control_variate": self.local_control_variate_plus,
        }
        with open(self.local_state_file, "wb") as f:
            pickle.dump(local_state, f)

    def load_local_state(self):
        if os.path.exists(self.local_state_file):
            with open(self.local_state_file, "rb") as f:
                local_state = pickle.load(f)
                self.local_control_variate = local_state["local_control_variate"]


class SCAFFOLD:
    """
    Federated Averaging algorithm.
    """

    def __init__(self, args, train_data, test_data):
        """
        Initialize SCAFFOLD.

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

        self.global_control_variate = {
            name: torch.zeros_like(param)
            for name, param in self.global_model.named_parameters()
        }

        self.directory_path = f"data/local_state/{wandb.run.id}"
        if os.path.exists(self.directory_path):
            shutil.rmtree(self.directory_path)
        os.makedirs(self.directory_path)

    def select_clients(self):
        """
        Randomly select a subset of clients for training.

        Returns:
            Sorted list of selected client IDs.
        """
        random.seed(self.args.seed * (self.round + 1))
        num_clients_to_select = self.args.selected_clients
        return sorted(random.sample(self.client_ids, num_clients_to_select))

    def aggregate(self, client_updates):

        total_loss = sum(
            [client_update["loss"] for client_update in client_updates]
        ) / len(client_updates)

        wandb.log(
            {"training_loss": total_loss},
            step=self.round,
        )

        aggregated_weight = self.global_model.state_dict()

        for layer in aggregated_weight.keys():
            mean_w_d = torch.mean(
                torch.stack(
                    [
                        client_update["update"][0][layer].float()
                        for client_update in client_updates
                    ]
                ),
                dim=0,
            )
            mean_c_d = torch.mean(
                torch.stack(
                    [
                        client_update["update"][1][layer].float()
                        for client_update in client_updates
                    ]
                ),
                dim=0,
            )

            aggregated_weight[layer] += self.global_lr * mean_w_d

            self.global_control_variate[layer] += (
                self.args.selected_clients / self.args.num_clients
            ) * mean_c_d

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

            arguments = [
                (
                    client_id,
                    copy.deepcopy(self.global_model.state_dict()),
                    self.train_data[client_id],
                    self.global_control_variate,
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

            self.aggregate(client_updates)

            evaluate(self.global_model, self.test_data, current_round, self.args)
        if os.path.exists(self.directory_path):
            shutil.rmtree(self.directory_path)
        os.makedirs(self.directory_path)

    def client_train(self, params):
        """
        Train a client and return its update.

        Parameters:
            params: Tuple containing (client_id, model_parameter, data_loader, args).

        Returns:
            Dictionary with client_id, update, and loss.
        """
        client_id, model_parameter, data_loader, global_control_variate, args = params
        client = SCAFFOLDClient(
            client_id, data_loader, model_parameter, global_control_variate, args
        )
        client.load_local_state()
        loss = client.train()

        update = client.get_update()

        return {
            "client_id": client_id,
            "update": update,
            "loss": loss,
        }
