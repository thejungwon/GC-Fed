# Gradient Centralized Federated Learning (GC-Fed)
<p align="center">
        <a href="https://arxiv.org/abs/2503.13180"><img src="https://img.shields.io/badge/arXiv-2503.13180-b31c1c"></a>
</p>

<p align="center">
    <img src="https://github.com/user-attachments/assets/e8d00e05-c6c4-4734-bb3d-e5188f0066d0" alt="GC-Fed Image">
</p>

>Multi-source information fusion (MSIF) leverages diverse data streams to enhance decision-making, situational awareness, and system resilience. Federated Learning (FL) enables MSIF while preserving privacy but suffers from client drift under high data heterogeneity, leading to performance degradation. Traditional mitigation strategies rely on reference-based gradient adjustments, which can be unstable in partial participation settings. To address this, we propose Gradient Centralized Federated Learning (GC-Fed), a reference-free gradient correction method inspired by Gradient Centralization (GC). We introduce Local GC and Global GC, applying GC during local training and global aggregation, respectively. Our hybrid GC-Fed approach selectively applies GC at the feature extraction layer locally and at the classifier layer globally, improving training stability and model performance. Theoretical analysis and empirical results demonstrate that GC-Fed mitigates client drift and achieves state-of-the-art accuracy gains of up to 20% in heterogeneous settings.




This repository contains the official PyTorch implementation of GC-Fed.

## Datasets

The following datasets are supported:

- EMNIST
- CIFAR-10
- CIFAR-100
- TinyImageNet

Dataset will be automatically downloaded for the first run. See `dataloader.py`.

## Models

Supported model architectures:

- MLP
- CNN
- VGG11
- ResNet18

See `models.py`.

## Baseline Algorithms
<p align="center">
    <img src="https://github.com/user-attachments/assets/acb90844-f338-4e70-bf12-956c950c947e" alt="Training Dynamics">
</p>

GC-Fed supports multiple federated learning baseline algorithms, including:

- FedAvg
- FedProx
- SCAFFOLD
- FedDyn
- FedNTD
- FedVARP
- FedLC
- FedDecorr
- FedSOL
- FedACG

See `algorithms/`.

## Tested Environments

GC-Fed has been tested in the following environments:

- **Operating Systems**: Ubuntu (CUDA 12.2), macOS (M1, MPS)
- **Python Version**: 3.9
- **Package Manager**: pip 25.0.1

## Installation

To set up the environment, run:

```sh
pip install -U pip
pip install -r requirements.txt
```

## Running the Code

To train a model using FedAvg, use the following command:

```sh
python main.py --algorithm fedavg --project_name GCFED \
    --lr 0.01 --momentum 0.9 --weight_decay 0.00001 \
    --batch_size 50 --epochs 5 --noniid 0.1 \
    --selected_clients 5 --rounds 800 --num_clients 100 \
    --seed 40 --model cnn --dataset cifar10 --send_gradients
```

For GC-Fed:

```sh
python main.py --algorithm gcfed --project_name GCFED \
    --lr 0.01 --momentum 0.9 --weight_decay 0.00001 \
    --batch_size 50 --epochs 5 --noniid 0.1 \
    --selected_clients 5 --rounds 800 --num_clients 100 \
    --seed 40 --model cnn --dataset cifar10 --send_gradients --gc_target_layer fc2
```

## Training Configuration

Key parameters:

- `--dataset`: Dataset name (e.g., `emnist`, `cifar10`, `cifar100`, `tinyimagenet`).
- `--model`: Backbone model (e.g., `mlp`, `cnn`, `vgg11`, `resnet18`).
- `--algorithm`: Federated learning baseline algorithm (e.g., `fedavg`, `fedprox`, etc., see `algorithms/__init__.py`).
- `--noniid`: Level of data heterogeneity (LDA alpha). Lower values indicate higher heterogeneity (e.g., `0.05` is highly heterogeneous, `1000` is nearly IID).
- `--num_clients`: Total number of clients.
- `--selected_clients`: Number of clients participating per round.
- `--rounds`: Total number of communication rounds.

## GC-Fed Specific Parameters

- `--gc_target_layer`: Specifies the target layer to be centralized on the server side. This layer is excluded from local training.
- `--gc_layer_lambda`: Defines the ratio (0.0-1.0) of local gradient centralization. This parameter is effective only if `--gc_target_layer` is not set (empty `""`).

## Notes

The code has been updated to improve readability and efficiency by removing unnecessary device changes. As a result, performance may differ slightly from reported results.

## Important Packages

GC-Fed relies on the following libraries:

- [PyTorch](https://github.com/pytorch/pytorch)
- [WandB](https://github.com/wandb/wandb)
- [FedLab](https://github.com/SMILELab-FL/FedLab)
- [Avalanche](https://github.com/ContinualAI/avalanche/)

## References

Additional relevant resources:
- [Gradient Centralization](https://github.com/Yonghongwei/Gradient-Centralization)
- [Flower](https://github.com/adap/flower)
- [FedNTD](https://github.com/Lee-Gihun/FedNTD)
- [FedSOL](https://github.com/Lee-Gihun/FedSOL)
- [FedACG](https://github.com/geehokim/FedACG)
- [FedDecorr](https://github.com/bytedance/FedDecorr)

For further details, please refer to the respective repositories.
