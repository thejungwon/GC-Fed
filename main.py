from algorithms import get_algorithm
from args import get_args
from dataloader import load_data
from utils import set_random_seed, setup_logging

import warnings

warnings.filterwarnings("ignore")
import torch
import wandb


def main():
    args = get_args()
    print(args)
    setup_logging(args)
    set_random_seed(args.seed)
    run_name = f"{args.algorithm}"

    wandb.init(project=args.project_name, name=run_name, mode="online")

    # Load data
    print("Data Loading!")
    train_data, test_data = load_data(args)

    args.num_clients = len(train_data)
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    args.device = device
    wandb.config.update(vars(args))

    # Initialize algorithm
    algorithm = get_algorithm(args.algorithm, args, train_data, test_data)

    # Start training
    algorithm.train()


if __name__ == "__main__":
    main()
