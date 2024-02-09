import torch


def get_mean_std(dataset):
    mean = [0.0, 0.0, 0.0]
    std = [0.0, 0.0, 0.0]

    for image, _ in dataset:
        for i in range(3):
            mean[i] += image[i, :, :].mean()
            std[i] += image[i, :, :].std()

    total_images = len(dataset)

    for i in range(3):
        mean[i] /= total_images
        std[i] /= total_images

    mean = torch.tensor(mean).tolist()
    std = torch.tensor(std).tolist()

    mean = [round(value, 1) for value in mean]
    std = [round(value, 1) for value in std]

    return mean, std
