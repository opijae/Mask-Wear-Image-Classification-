import argparse
import glob
import json
import os
import random
import re
from importlib import import_module
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from adamp import AdamP

from dataset import MaskBaseDataset,AgeDataset,AgeDataset_1
from loss import create_criterion
from scheduler import create_scheduler
# from age import AgeDataset


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    # figure = plt.figure(figsize=(8, 12))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    # plt.subplots_adjust(top=0.6)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    scaler = torch.cuda.amp.GradScaler() 

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: BaseAugmentation
    dataset = dataset_module(
        data_dir=data_dir,
        val_ratio=args.val_ratio
    )

    dataset_module = getattr(import_module("dataset"),'AgeDataset')  # default: BaseAugmentation
    dataset1 = dataset_module(
        csv_file='/opt/ml/pstage_01_image_classification/appa-real-release/gt_train.csv',
        data_dir='/opt/ml/pstage_01_image_classification/appa-real-release/train',
        val_ratio=args.val_ratio
    )
    # num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)
    transform_module = getattr(import_module("dataset"), 'age_transform')
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset1.set_transform(transform)
    print(len(dataset))
    print(len(dataset1))
    # -- data_loader
    train_set, val_set = dataset.split_dataset()
    train_set1, val_set1 = dataset1.split_dataset()
    list_of_train_datasets=[train_set,train_set1]
    list_of_valid_datasets=[val_set,val_set1]
    # list_of_datasets.
    train_set = data.ConcatDataset(list_of_train_datasets)
    val_set = data.ConcatDataset(list_of_valid_datasets)
    print(len(train_set))
    print(len(val_set))
    # print(multiple_dataset[1658])
    # exit()
    # print(train_set[0])
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )
    val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            num_workers=8,
            shuffle=True,
            pin_memory=use_cuda,
            drop_last=True,
        )
    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(
        # num_classes=num_classes   
    ).to(device)
    model = torch.nn.DataParallel(model)
    # print(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    if args.optimizer=='Adamp':
        optimizer = AdamP(
            filter(lambda p: p.requires_grad, 
            model.parameters()), 
            lr=args.lr, 
            betas=(0.9, 0.999), 
            weight_decay=1e-2)
    else:
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=5e-4
        )
    if args.scheduler=='step_lr':
        # lr_decay_step=int(input('lr_decay_step : '))
        # scheduler=create_scheduler(args.scheduler,optimizer,lr_decay_step)
        scheduler = StepLR(optimizer, 5, gamma=0.5)
    else:
        scheduler=create_scheduler(args.scheduler,optimizer)
    # scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    
    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_loss = np.inf
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)

            # print(labels)
            optimizer.zero_grad()

            outs = model(inputs)
            loss = torch.sqrt(criterion(outs, labels))
            loss.backward()
            optimizer.step()
            loss_value += loss.item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} ||"
                )
                # logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                # logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0
        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            # figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.float().to(device)
                labels = labels.float().to(device)

                outs = model(inputs)
                # preds = torch.argmax(outs, dim=-1)

                # loss_item = criterion(outs, labels).item()
                loss_item = torch.sqrt(criterion(outs, labels)).item()
                val_loss_items.append(loss_item)
                print(f'label {labels} || pred {outs}')
            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_loss = min(best_loss, val_loss)
            if best_loss>val_loss:
                print(f"New best model for val accuracy : {val_loss:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_loss=val_loss
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            val_loss=0
        scheduler.step()
        
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--scheduler', type=str, default='step_lr', help='scheculer (default: step_lr)')

    # parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--is_multi', type=str, default='n', help='multi_lable?')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model_2'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)