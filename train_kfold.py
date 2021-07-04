import argparse
import glob
import json
import os
import random
import re
from importlib import import_module
from pathlib import Path
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from adamp import AdamP
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from dataset import MaskBaseDataset ,TestDataset
from loss import create_criterion
from scheduler import create_scheduler

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

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: BaseAugmentation
    dataset = dataset_module(
        data_dir=data_dir,
        val_ratio=args.val_ratio
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)


    
    
    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf


    n_splits = 5
    # skf = StratifiedKFold(n_splits=n_splits)
    kfold = KFold(n_splits=n_splits, shuffle=False)
    oof_pred = None

    ############ test data loader


    img_root = os.path.join('/opt/ml/input/data/eval', 'images')
    info_path = os.path.join('/opt/ml/input/data/eval', 'info.csv')
    info = pd.read_csv(info_path)
    # print(info)
    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]

    test_dataset = TestDataset(img_paths, args.resize)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )


    for i, (train_idx, valid_idx) in enumerate(kfold.split(dataset)):

        print(train_idx, valid_idx)
        # continue
        # train_loader, val_loader = getDataloader(dataset, train_idx, valid_idx, batch_size, num_workers)

        train_set = torch.utils.data.Subset(dataset,
                                            indices=train_idx)
        val_set   = torch.utils.data.Subset(dataset,
                                            indices=valid_idx)
        # val_set.dataset.set_phase("test")
        # continue
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
        model_module = getattr(import_module("model"), args.model)  # default: BaseModel
        model = model_module(
            num_classes=num_classes
        ).to(device)

        model = torch.nn.DataParallel(model)
        criterion = create_criterion(args.criterion)
        # train_params = [{'params': getattr(model.net, 'features').parameters(), 'lr': args.lr / 10, 'weight_decay':5e-4},
        #                 {'params': getattr(model.net, 'fc').parameters(), 'lr': args.lr, 'weight_decay':5e-4}]

        # optimizer = AdamP(train_params)
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

        for epoch in range(args.epochs):
            # train loop
            model.train()
            loss_value = 0
            matches = 0
            for idx, train_batch in enumerate(train_loader):
                inputs, labels = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                
                # print(preds)
                loss = criterion(outs, labels)

                loss.backward()
                optimizer.step()
                # if (idx+1) % 2 == 0:
                #     optimizer.step()
                #     optimizer.zero_grad()
                loss_value += loss.item()
                matches += (preds == labels).sum().item()
                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    current_lr = get_lr(optimizer)
                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                    )
                    logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                    logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                    loss_value = 0
                    matches = 0
            scheduler.step()

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                # figure = None
                for val_batch in val_loader:
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)

                    loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(val_set)
                best_val_loss = min(best_val_loss, val_loss)
                if val_acc > best_val_acc:
                    print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                    torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                    best_val_acc = val_acc
                torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
                print(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                )
                logger.add_scalar("Val/loss", val_loss, epoch)
                logger.add_scalar("Val/accuracy", val_acc, epoch)
                try:
                    logger.add_figure("results", figure, epoch)
                except:
                    pass
                print()
        all_predictions = []
        with torch.no_grad():
            print(f"{i} Calculating test results...")
            for images in test_loader:
                images = images.to(device)

                # Test Time Augmentation
                pred = model(images) /2# 원본 이미지를 예측하고
                pred += model(torch.fliplr(images))/2 # flip으로 뒤집어 예측합니다. 

                # pred_softmax=pred.softmax(-1)
                all_predictions.extend(pred.cpu().numpy())

            fold_pred = np.array(all_predictions)
            # pred_softmax_numpy=pred_softmax.cpu().numpy()
            # np.savetxt(f'softmax_{i}.csv',pred_softmax_numpy,fmt='%f',delimiter=',')
            try:
                np.savetxt(f'file_all_predictions_{args.model}_{args.lr}_{i}.csv',all_predictions,fmt='%f',delimiter=',')
            except:
                np.savetxt(f'file_middle_{i}.csv',fold_pred,fmt='%f',delimiter=',')
        if oof_pred is None:
            oof_pred = fold_pred / n_splits
            # oof_pred = pred_softmax_numpy
        else:
            oof_pred += fold_pred / n_splits
            # oof_pred *= pred_softmax_numpy
        
        np.savetxt(f'check_{args.model}_{i}.csv',oof_pred,fmt='%f',delimiter=',')
    np.savetxt(f'final_{args.model}_{args.lr}.csv',oof_pred,fmt='%f',delimiter=',')
    # oof_pred=oof_pred**(1/n_splits)
    info['ans'] = np.argmax(oof_pred, axis=1)
    info.to_csv(os.path.join('./output', f'output_{args.model}.csv'), index=False)
    # submission.to_csv(os.path.join(test_img_root, 'submission.csv'), index=False)
    print('test inference is done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[224, 224], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=64, help='input batch size for validing (default: 1000)')
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
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)