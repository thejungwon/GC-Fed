import argparse
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser()

    # General Parameter
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Directory for storing data"
    )
    parser.add_argument("--model", type=str, default="cnn", help="Model type")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset to use")
    parser.add_argument(
        "--algorithm", type=str, default="fedavg", help="Federated learning algorithm"
    )

    # Parameters for local training
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Local training learning rate"
    )
    parser.add_argument(
        "--lr_decay", type=float, default=1.0, help="Learning rate decay per round"
    )
    parser.add_argument("--momentum", type=float, default=0.0, help="Momentum")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")

    parser.add_argument(
        "--batch_size", type=int, default=50, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of local training epochs"
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="cross-entropy",
        help="criterion for local training",
    )

    # Parameters for federated learning
    parser.add_argument(
        "--noniid", type=float, default=0.1, help="Degree of non-IID (LDA)"
    )

    parser.add_argument(
        "--num_clients", type=int, default=100, help="Number of total clients"
    )
    parser.add_argument(
        "--selected_clients", type=int, default=5, help="Number of selected clients"
    )
    parser.add_argument(
        "--rounds", type=int, default=100, help="Number of global training rounds"
    )
    parser.add_argument(
        "--send_gradients",
        action="store_true",
        help="Send gradients (update) instead of weights",
    )

    # Parameters for Wandb
    parser.add_argument(
        "--project_name", type=str, default="GCFED", help="Project Name"
    )
    parser.add_argument("--desc", type=str, default="default", help="Description")

    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device for computing, will be automatically detected.",
    )

    parser.add_argument(
        "--log_file", type=str, default="training.log", help="Log file path"
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # FL algorithm specific parameter

    parser.add_argument(
        "--feddyn_alpha", type=float, default=0.1, help="alpha for feddyn"
    )
    parser.add_argument(
        "--fedacg_beta",
        type=float,
        default=0.001,
        help="proximal loss weight for fedacg",
    )
    parser.add_argument(
        "--fedacg_server_momentum",
        type=float,
        default=0.85,
        help="Server momentum for fedacg",
    )

    parser.add_argument(
        "--feddecorr_coef",
        type=float,
        default=0.001,
        help="proximal loss weight for feddecor",
    )
    parser.add_argument(
        "--fedlc_calibration_temp",
        type=float,
        default=0.1,
        help="Calibration strength for fedlc",
    )

    parser.add_argument("--fedprox_mu", type=float, default=0.1, help="mu for fedprox")

    parser.add_argument("--fedsol_rho", type=float, default=1.0, help="rho for fedsol")

    parser.add_argument("--fedntd_tau", type=float, default=1.0, help="tau for fedntd")
    parser.add_argument(
        "--fedntd_beta", type=float, default=1.0, help="beta for fedntd"
    )

    parser.add_argument(
        "--gc_target_layer", type=str, default="", help="layer for global projection"
    )
    parser.add_argument(
        "--gc_layer_lambda", type=float, default=0.9, help="ratio of local projection"
    )

    return parser.parse_args()
