import argparse
import os
from importlib import import_module
from efficientnet_pytorch import EfficientNet
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import TestDataset, MaskBaseDataset,TestGrayDataset

def load_model(saved_model,  device,is_age):
    
    if is_age:
        model_cls = getattr(import_module("model"), args.model)
        model = model_cls()
    else:
        model_cls = getattr(import_module("model"),'efficientnet_b4')
        model = model_cls(num_classes=18)

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model
@torch.no_grad()
def inference_only_age(data_dir,model_dir,args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = load_model(model_dir, device,True).to(device)
    model.eval()

    # img_root = os.path.join(data_dir, 'images')
    # info_path = os.path.join(data_dir, 'info.csv')
    # info = pd.read_csv(info_path)

    # img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    data_list=os.listdir(data_dir)
    temp=[os.path.join(data_dir,x) for x in data_list]
    dataset = TestDataset(temp, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )
    print("Calculating inference results..")
    preds = []

    with torch.no_grad():
        for idx, images in enumerate(loader):
            images=images.float().to(device)
            # print(images[0])
            outs = model(images)
            print(outs)
@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = 18  # 18
    temp=args.model
    # model = load_model('./', device,False).to(device)
    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=18)
    model.load_state_dict(torch.load('best.pth', map_location=device))
    model.to(device)
    # print(model)
    model.eval()
    age_model = load_model('./model_2/exp11', device,True).to(device)
    age_model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths,0)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    dataset1 = TestDataset(img_paths, args.resize)
    loader1 = torch.utils.data.DataLoader(
        dataset1,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    age_preds=[]
    temp=[]
    cnt=0
    label=[0]
    with torch.no_grad():
        for idx, images in enumerate(loader):            
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())
        np.savetxt(f'pure_predict.csv',preds,fmt='%f',delimiter=',')
        print('done1')
        for idx, images in enumerate(loader1):
            images = images.to(device)
            age_outs=age_model(images)
            age_label=age_outs.cpu().numpy()
            temp.extend(age_label)
            age_label=np.where(age_label>52,2,(np.where(age_label<30,0,1)))
            age_preds.extend(age_label)
        np.savetxt(f'age_predict.csv',temp,fmt='%f',delimiter=',')
        print('done2')
    for p,a in zip(preds,age_preds):
        if p%3!=a:
            cnt+=1
        label.append(p-(p%3-a))
    # np.savetxt(f'outputs/pred_53.csv',preds,fmt='%f',delimiter=',')
    # np.savetxt(f'outputs/age_55_25.csv',age_preds,fmt='%f',delimiter=',')
    label=list(map(int,label))
    np.savetxt(f'outputs/final_52.csv',label,fmt='%f',delimiter=',')
    # info['ans'] = preds
    # info.to_csv(os.path.join(output_dir, f'output_s.csv'), index=False)
    # print(f'Inference Done!')
    print(cnt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(128, 96), help='resize size for image when you trained (default: (96, 128))')
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
    # inference_only_age(data_dir, model_dir,  args)

