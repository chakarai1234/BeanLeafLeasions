import torch


def get_model(pretrained_model, model_weights=None, out_features: int = 0):

    model = pretrained_model(weights=model_weights)

    if list(model.named_children())[-1][0] == 'fc':
        num_ftrs = model.fc.in_features

        # model.fc = torch.nn.Linear(
        #     in_features=num_ftrs, out_features=out_features)

        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(
                in_features=num_ftrs, out_features=out_features)
        )
    elif list(model.named_children())[-1][0] == 'classifier':
        num_ftrs = model.classifier[-1].in_features
        # model.classifier[-1] = torch.nn.Linear(
        #     in_features=num_ftrs, out_features=out_features)

        model.classifier[-1] = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(
                in_features=num_ftrs, out_features=out_features)
        )

    return model
