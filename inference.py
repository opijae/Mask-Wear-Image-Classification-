import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset,TestGrayDataset


def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

# def pred_model(model,):


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = args.num_classes  # 18
    temp=args.model
    if args.is_seperate=='y':
        num_classes_list=[3,2,3]
        model_list=[]
        for idx,num_classes in enumerate(num_classes_list):
            model_dir=input('mask_model_dir: ')
            if idx==0 and args.is_gray=='y':
                args.model+='_gray'
            model = load_model(model_dir, num_classes, device).to(device)
            model.eval()
            model_list.append(model)
            args.model=temp
    else:
        model = load_model(model_dir, num_classes, device).to(device)
        model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )
    # gray_dataset=TestGrayDataset(img_paths, args.resize)
    # gray_loader = torch.utils.data.DataLoader(
    #     gray_dataset,
    #     batch_size=args.batch_size,
    #     num_workers=8,
    #     shuffle=False,
    #     pin_memory=use_cuda,
    #     drop_last=False,
    # )   

    print("Calculating inference results..")
    preds = []

    with torch.no_grad():
        # if args.is_gray=='y':

        for idx, images in enumerate(loader):
            label=0
            if args.is_seperate=='y':
                for idx_2,model in enumerate(model_list):
                    if args.is_gray=='y' and idx_2==0:
                        temp=images
                        images=gray_images
                    images = images.to(device)
                    pred = model(images)
                    pred = pred.argmax(dim=-1)
                    if args.is_gray=='y':
                        images=temp
                    if idx_2==0:
                        pred = pred*6
                    elif idx_2==1:
                        pred = pred*3
                    label+=pred
            else:
                images = images.to(device)
                outs = model(images)
                # pred = pred.argmax(dim=-1)
                if args.is_multi=='y':
                    index=torch.argsort(outs)[:,:3]
                    mask_index=torch.argmax(outs[:,:3],dim=-1)
                    gender_index=torch.argmax(outs[:,3:5],dim=-1)
                    age_index=torch.argmax(outs[:,5:],dim=-1)
                    label=mask_index*6+gender_index*3+age_index
                    # preds=torch.zeros_like(outs).scatter_(1,index,torch.ones_like(outs))
                    # print(index.shape)
                    # index,_=index.sort(dim=1)
                    # print(index.shape)
                    # temp=[]
                    # for idx in index:
                    #     # print(idx)
                    #     label=idx[0]*6+(idx[1]-3)*3+idx[2]-5
                    #     temp.append(label.cpu().numpy())
                    # # print(temp)
                    # label=temp
                    # print(len(temp))
                    # label=torch.tensor(temp)
                else:
                    pred = torch.argmax(outs, dim=-1)
                    label=pred
            preds.extend(label.cpu().numpy())
            # preds.extend(label)

    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(224, 224), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--num_classes', type=int, default='18', help='num_classes (default: 18)')
    parser.add_argument('--is_seperate', type=str, default='n', help='split(mask,gender,age) (default: n)')
    parser.add_argument('--is_gray', type=str, default='n', help='gray_channel (default: n)')
    parser.add_argument('--is_multi', type=str, default='n', help='multi_lable?')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
