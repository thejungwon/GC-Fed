import torch.nn as nn
import torch.nn.init as init
from torchvision.models import resnet18, vgg11
from functools import partial
import torch


def init_weights(model, init_type, init_gain):
    """Function for initializing network weights.

    Args:
        model: A torch.nn instance to be initialized.
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal).
        init_gain: Scaling factor for (normal | xavier | orthogonal).

    Reference:
        https://github.com/DS3Lab/forest-prediction/blob/master/pix2pix/models/networks.py
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            else:
                raise NotImplementedError(
                    f"[ERROR] ...initialization method [{init_type}] is not implemented!"
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif (
            classname.find("BatchNorm2d") != -1
            or classname.find("InstanceNorm2d") != -1
        ):
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    model.apply(init_func)


def get_model(args):
    model_name = args.model
    if model_name == "cnn":
        input_channels, num_classes, image_size = (
            args.input_channels,
            args.num_classes,
            args.image_size,
        )

        model = CNN(input_channels, num_classes, image_size)
        init_weights(model, "kaiming", 1.0)

        return model

    elif model_name == "mlp":

        model = MLP()
        init_weights(model, "kaiming", 1.0)

        return model

    elif model_name == "resnet18":
        model = ResNet18(args.num_classes, args.image_size)

        return model

    elif model_name == "vgg11":
        model = VGG11(args.num_classes)

        return model
    else:
        raise ValueError(f"Model {model_name} not supported")


class CNN(nn.Module):
    def __init__(self, input_channels, num_classes, image_size):
        super(CNN, self).__init__()
        # Define a simpler CNN architecture
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()

        # Calculate the size of the feature map after the convolutions and pooling
        self.feature_map_size = self._calculate_feature_map_size(image_size)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * self.feature_map_size * self.feature_map_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def _calculate_feature_map_size(self, image_size):
        # Helper function to calculate the output size after convolutions and pooling
        size = image_size
        size = (size - 5 + 2 * 2) // 1 + 1  # After conv1
        size = size // 2  # After first pooling
        size = (size - 5 + 2 * 2) // 1 + 1  # After conv2
        size = size // 2  # After second pooling
        return size

    def forward(self, x, return_features=False):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        features = x.view(x.size(0), -1)  # Flatten the feature map
        x = self.relu(self.fc1(features))
        x = self.fc2(x)
        if return_features:
            return x, features
        else:
            return x


class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[512, 256], num_classes=62):
        super(MLP, self).__init__()
        # Define the layers for the MLP
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, return_features=False):
        x = x.view(x.size(0), -1)  # Flatten the image
        x = self.relu(self.fc1(x))
        features = self.relu(self.fc2(x))
        output = self.fc3(features)
        if return_features:
            return output, features
        else:
            return output


class ResNet18(nn.Module):
    def __init__(self, num_classes, image_size=None):
        super(ResNet18, self).__init__()
        num_groups = 32  # You can adjust this value as needed
        norm_layer = partial(nn.GroupNorm, num_groups)
        self.model = resnet18(
            pretrained=False,
            norm_layer=norm_layer,
        )

        # For lower scale image
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.model.maxpool = nn.Identity()

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x, return_features=False):
        # Pass input through the layers up to the avgpool layer
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        features = torch.flatten(x, 1)
        out = self.model.fc(features)
        if return_features:
            return out, features  # Return the features before the fc layer
        return out


class VGG11(nn.Module):
    def __init__(self, num_classes):
        super(VGG11, self).__init__()

        # Load the VGG11 architecture
        self.model = vgg11(pretrained=False)

        # Change the classifier's final layer to match the number of output classes
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x, return_features=False):
        # Process input through the feature extractor
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        if return_features:
            # Obtain features from the classifier excluding the final layer
            features = self.model.classifier[:-1](x)
            out = self.model.classifier[-1](features)
            return out, features
        else:
            return self.model.classifier(x)
