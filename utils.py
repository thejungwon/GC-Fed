import logging
import numpy as np
import torch
import random
import torch.nn as nn

# utils.py

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


def setup_logging(args):
    logging.basicConfig(filename=args.log_file, level=logging.INFO)


def log_info(message):
    logging.info(message)


def set_random_seed(seed=42):
    # Setting the random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Ensuring reproducibility in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import wandb


def evaluate(global_model, test_data, round, args):
    global_model.eval()
    correct_top1 = 0
    total = 0
    loss = 0
    criterion = nn.CrossEntropyLoss(reduction="sum").to(args.device)
    global_model = global_model.to(args.device)

    with torch.no_grad():
        for inputs, labels in test_data:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = global_model(inputs)
            loss += criterion(outputs, labels)

            _, predicted_top1 = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct_top1 += (predicted_top1 == labels).sum().item()

    accuracy_top1 = 100 * correct_top1 / total

    loss = loss / total

    log_info(
        f"Top-1 Accuracy of the global model on the test dataset: {accuracy_top1:.3f}%"
    )

    log_info(f"Loss of the global model on the test dataset: {loss:.3f}")

    wandb.log(
        {"accuracy": accuracy_top1, "loss": loss},
        step=round,
    )

    if torch.isnan(loss):
        log_info("Loss became NaN. Exiting evaluation.")
        exit()
