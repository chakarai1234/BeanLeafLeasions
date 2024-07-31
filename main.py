import os
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.models as models
from torch.nn import CrossEntropyLoss
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from argparse import ArgumentParser, Namespace
from torchinfo import Verbosity, summary
from datetime import datetime
from utils import *
import matplotlib.pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import make_grid
import shutil
import numpy as np
from pathlib import Path


def train_validate_test(arguments: ArgumentParser):

    args: Namespace = arguments.parse_args()

    RESIZE: set[int] = args.resize
    mean: list[float] = [0.5]
    std: list[float] = [0.5]
    BATCH_SIZE: int = args.batch_size
    LEARNING_RATE: float = args.learning_rate
    WEIGHT_DECAY: float = args.weight_decay
    EPOCHS: int = args.epochs
    OUT_FEATURES: int = args.out_features
    LOG_SAVE_DIR: str = args.log_path
    IMG_SAVE_DIR: str = args.img_path
    MODEL_SAVE_DIR: str = args.model_path
    SAVE_MODEL: bool = args.save_model
    TENSORBOARD_LOGS_DIR: str = args.tensorboard_logs
    LJUST_VALUES: float = 20

    time_str = datetime.strftime(datetime.now(), '%m%d-%H%M%S')

    IMG_SAVE_DIR = IMG_SAVE_DIR+"/train-{:s}".format(time_str)

    torch.manual_seed(seed=42)

    Path(LOG_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    Path(IMG_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    if os.path.exists(TENSORBOARD_LOGS_DIR):
        shutil.rmtree(Path(TENSORBOARD_LOGS_DIR))
    Path(TENSORBOARD_LOGS_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)

    # os.path.join(LOG_SAVE_DIR, 'train-{:s}.log'.format(time_str))
    logger = get_logger(logger_name="Bean Leaf Lesions")

    DEVICE = get_device(args.device)

    model = get_model(models.resnet50, models.ResNet50_Weights.DEFAULT, OUT_FEATURES)

    model_details = summary(model, depth=20, batch_dim=BATCH_SIZE, device=DEVICE, verbose=Verbosity.QUIET)

    logger.info(f"\n{model_details}\n")

    log(args, logger, LJUST_VALUES)

    total_dataset = ImageFolder(args.train_path, transform=transforms.ToTensor())

    mean, std = get_mean_std(total_dataset)

    logger.info("{}:\t{}".format("mean".ljust(LJUST_VALUES), mean))
    logger.info("{}:\t{}".format("std".ljust(LJUST_VALUES), std))

    random_numbers = np.random.rand(4)

    scaled_numbers = random_numbers / np.sum(random_numbers) * 0.5

    brightness, contrast, saturation, hue = scaled_numbers

    brightness = round(brightness, 2)
    contrast = round(contrast, 2)
    saturation = round(saturation, 2)
    hue = round(hue, 2)

    logger.info("{}:\t{}".format("brightness".ljust(LJUST_VALUES), brightness))
    logger.info("{}:\t{}".format("contrast".ljust(LJUST_VALUES), contrast))
    logger.info("{}:\t{}".format("saturation".ljust(LJUST_VALUES), saturation))
    logger.info("{}:\t{}".format("hue".ljust(LJUST_VALUES), hue))

    train_transform = transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(args.train_path)
    test_dataset = ImageFolder(args.test_path, transform=test_transform)

    logger.info("{}:\t{}".format("Classes".ljust(LJUST_VALUES), dataset.class_to_idx))

    # train_dataset, val_dataset = random_split(dataset=dataset, lengths=(0.7, 0.3))

    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    class CustomDataset(Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            image, label = self.subset[idx]
            if self.transform:
                image = self.transform(image)
            return image, label

    train_dataset = CustomDataset(train_dataset, transform=train_transform)
    val_dataset = CustomDataset(val_dataset, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    logger.info("{}:\t{}".format("Train Loader length".ljust(LJUST_VALUES), len(train_loader)))
    logger.info("{}:\t{}".format("Val Loader length".ljust(LJUST_VALUES), len(val_loader)))
    logger.info("{}:\t{}".format("Test Loader length".ljust(LJUST_VALUES), len(test_loader)))

    model.to(DEVICE)

    criterion = CrossEntropyLoss().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=EPOCHS)

    start_time = str(datetime.now().strftime("%H:%M:%S"))
    t1 = datetime.strptime(start_time, "%H:%M:%S")
    logger.info("{}:\t{}".format("Start time".ljust(LJUST_VALUES), t1.time()))

    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    last_lrs = []

    # writer = SummaryWriter(TENSORBOARD_LOGS_DIR, comment="Resnet50")

    # for epoch in range(EPOCHS):
    #     model.train()
    #     total_loss = 0.0
    #     correct = 0
    #     total = 0

    #     for i, (images, labels) in enumerate(train_loader):

    #         images, labels = images.to(DEVICE), labels.to(DEVICE)

    #         optimizer.zero_grad()
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs, 1)
    #         loss = criterion(outputs, labels)

    #         loss.backward()
    #         optimizer.step()

    #         correct += (predicted == labels).sum().item()
    #         total += labels.size(0)

    #         total_loss += loss.item()

    #         grid = make_grid(images)
    #         writer.add_image(f'Training_Epoch_{epoch+1}', grid, i)
    #         writer.close()

    #     correct_val = 0
    #     total_val = 0
    #     total_loss_val = 0.0

    #     model.eval()

    #     with torch.inference_mode():
    #         for i, (images, labels) in enumerate(val_loader):
    #             images, labels = images.to(DEVICE), labels.to(DEVICE)
    #             outputs = model(images)
    #             loss = criterion(outputs, labels)
    #             _, predicted = torch.max(outputs, 1)
    #             total_val += labels.size(0)
    #             correct_val += (predicted == labels).sum().item()

    #             total_loss_val += loss.item()

    #             grid = make_grid(images)
    #             writer.add_image(f'Val_Epoch_{epoch+1}', grid, i)
    #             writer.close()

    #     scheduler.step()

    #     train_accuracy = round(100*correct/total, 2)
    #     val_accuracy = round(100*correct_val/total_val, 2)
    #     train_loss = round(total_loss/len(train_loader), 2)
    #     val_loss = round(total_loss_val/len(val_loader), 2)
    #     last_lr = scheduler.get_last_lr()[0]

    #     train_accuracies.append(train_accuracy)
    #     val_accuracies.append(val_accuracy)
    #     train_losses.append(train_loss)
    #     val_losses.append(val_loss)
    #     last_lrs.append(last_lr)

    #     writer.add_scalar('Loss/Train', train_loss, epoch)
    #     writer.add_scalar('Loss/Val', val_loss, epoch)
    #     writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
    #     writer.add_scalar('Accuracy/Val', val_accuracy, epoch)
    #     writer.add_scalar('Learning Rate', last_lr, epoch)

    #     writer.close()

    #     logger.info(
    #         "{}:\tTrain Accuracy: {}% - Val Accuracy: {}% - Train loss: {} - Val loss: {} - Last LR: {}"
    #         .format(f"Epoch {epoch+1}".ljust(LJUST_VALUES),
    #                 f"{train_accuracy}",
    #                 f"{val_accuracy}",
    #                 f"{train_loss}",
    #                 f"{val_loss}",
    #                 f"{last_lr}"))

    # logger.info("{}:\t{}".format("Train Accuracies".ljust(LJUST_VALUES), train_accuracies))
    # logger.info("{}:\t{}".format("Val Accuracies".ljust(LJUST_VALUES), val_accuracies))
    # logger.info("{}:\t{}".format("Train Losses".ljust(LJUST_VALUES), train_losses))
    # logger.info("{}:\t{}".format("Val Losses".ljust(LJUST_VALUES), val_losses))
    # logger.info("{}:\t{}".format("Last Learning Rates".ljust(LJUST_VALUES), last_lrs))

    # plt.plot(range(EPOCHS), train_accuracies, label="Train Accuracies")
    # plt.plot(range(EPOCHS), val_accuracies, label="Val Accuracies")
    # plt.legend()
    # plt.savefig(IMG_SAVE_DIR+"/accuracies-plot.png")
    # logger.info("Saving the accuracies plot")
    # plt.close()

    # plt.plot(range(EPOCHS), train_losses, label="Train Losses")
    # plt.plot(range(EPOCHS), val_losses, label="Val Losses")
    # plt.legend()
    # plt.savefig(IMG_SAVE_DIR+"/losses-plot.png")
    # logger.info("Saving the losses plot")
    # plt.close()

    # plt.plot(range(EPOCHS), last_lrs, label="Learning Rates")
    # plt.savefig(IMG_SAVE_DIR+"/learning-rate-plot.png")
    # logger.info("Saving the learning rate plot")
    # plt.legend()
    # plt.close()

    # correct_test = 0
    # total_test = 0
    # total_loss_test = 0.0

    # model.eval()

    # with torch.inference_mode():
    #     for images, labels in test_loader:
    #         images, labels = images.to(DEVICE), labels.to(DEVICE)
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)
    #         _, predicted = torch.max(outputs, 1)
    #         total_test += labels.size(0)
    #         correct_test += (predicted == labels).sum().item()

    #         total_loss_test += loss.item()

    #         grid = make_grid(images)
    #         writer.add_image(f'Test_Epoch', grid)
    #         writer.close()

    # logger.info("{}:\t{}".format(f"Test".ljust(LJUST_VALUES), f"Test Accuracy - {100*correct_test/total_test:.2f}% - Test Loss = {total_loss_test/len(test_loader):.3f}"))

    # end = str(datetime.now().strftime("%H:%M:%S"))
    # t2 = datetime.strptime(end, "%H:%M:%S")
    # logger.info("{}:\t{}".format("End time".ljust(LJUST_VALUES), t2.time()))

    # delta = t2 - t1

    # logger.info("{}:\t{}".format("Time Difference".ljust(LJUST_VALUES), delta))

    # if SAVE_MODEL:
    #     torch.save(model, MODEL_SAVE_DIR+"/model-{:s}.pth".format(time_str))
    #     logger.info("Saving the Model")
    # else:
    #     logger.info("Not saving the Model, Please set the --save-model flag")


if __name__ == "__main__":

    argparser = ArgumentParser(prog='Pretrained model training', description='Train the model with pretrained weights')

    argparser.add_argument("--device", type=str, default="mps", choices=["cpu", "cuda", "mps"], help="Device")

    argparser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    argparser.add_argument("--learning-rate", type=float, default=0.00001, help="Learning Rate")

    argparser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight Decay")

    argparser.add_argument("--epochs", type=int, default=20, help="Epochs")

    argparser.add_argument("--train-path", type=str, default=os.getcwd()+"/Data/train", help="Train path")

    argparser.add_argument("--test-path", type=str, default=os.getcwd()+"/Data/val", help="Test Path")

    argparser.add_argument("--log-path", type=str, default=os.getcwd()+"/logs", help="Log Path")

    argparser.add_argument("--img-path", type=str, default=os.getcwd()+"/images", help="Image Path")

    argparser.add_argument("--model-path", type=str, default=os.getcwd()+"/models", help="Model Path")

    argparser.add_argument("--tensorboard-logs", type=str, default=os.getcwd()+"/tensorboard_logs", help="Tensorboard logs Path")

    argparser.add_argument("--save-model", type=bool, default=False, help="Save Model")

    argparser.add_argument("--out-features", type=int, default=3, help="Output Features")

    argparser.add_argument("--resize", type=set, default=(224, 224), help="Output Features")

    train_validate_test(arguments=argparser)
