import torch
from torchvision.models._api import WeightsEnum
from typing import Union


def get_model(pretrained_model, model_weights: Union[WeightsEnum, None] = None, out_features: int = 0, freeze_until_custom_layer: bool = True):

    model = pretrained_model(weights=model_weights)

    if list(model.named_children())[-1][0] == 'fc':
        num_ftrs = model.fc.in_features

        # model.fc = torch.nn.Linear(
        #     in_features=num_ftrs, out_features=out_features)

        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(
                in_features=num_ftrs, out_features=out_features)
        )
    elif list(model.named_children())[-1][0] == 'classifier':
        num_ftrs = model.classifier[-1].in_features
        # model.classifier[-1] = torch.nn.Linear(
        #     in_features=num_ftrs, out_features=out_features)

        model.classifier[-1] = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(
                in_features=num_ftrs, out_features=out_features)
        )

        # Freeze layers until the custom layer
    if freeze_until_custom_layer:
        for name, param in model.named_parameters():
            param.requires_grad = False
        # Unfreeze the custom layers
        for name, param in model.named_parameters():
            if 'fc' in name or 'classifier' in name:
                param.requires_grad = True

    return model
