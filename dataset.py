import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.transforms import *
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from autoaugment import ImageNetPolicy
IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            CenterCrop((380, 380)),
            Resize(resize, Image.BILINEAR),
            # ColorJitter(0.1, 0.1, 0.1, 0.1),
            # ImageNetPolicy(), 
            ToTensor(),            
            Normalize(mean=mean, std=std),
            # AddGaussianNoise()
        ])

    def __call__(self, image):
        return self.transform(image)

class age_transform:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            # ColorJitter(0.1, 0.1, 0.1, 0.1),
            # ImageNetPolicy(), 
            ToTensor(),            
            Normalize(mean=mean, std=std),
            # AddGaussianNoise()
        ])

    def __call__(self, image):
        return self.transform(image)
class GrayAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            CenterCrop((320, 256)),
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            # Normalize(mean=mean, std=std),
            # AddGaussianNoise()
        ])

    def __call__(self, image):
        return self.transform(image)

class MaskBaseDataset(data.Dataset):
    num_classes = 3 * 2 * 3

    class MaskLabels:
        mask = 0
        incorrect = 1
        normal = 2

    class GenderLabels:
        male = 0
        female = 1

    class AgeGroup:
        map_label = lambda x: 0 if int(x) < 30 else 1 if int(x) < 58 else 2

    _file_names = {
        "mask1": MaskLabels.mask,
        "mask2": MaskLabels.mask,
        "mask3": MaskLabels.mask,
        "mask4": MaskLabels.mask,
        "mask5": MaskLabels.mask,
        "incorrect_mask": MaskLabels.incorrect,
        "normal": MaskLabels.normal
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = getattr(self.GenderLabels, gender)
                age_label = self.AgeGroup.map_label(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can takes huge amounts of time depending on your CPU machine :(")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)
        image_transform = self.transform(image)
        return image_transform, multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index):
        return self.mask_labels[index]

    def get_gender_label(self, index):
        return self.gender_labels[index]

    def get_age_label(self, index):
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label):
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label):
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8) # 값을 0에서 255로 제한
        return img_cp

    def split_dataset(self):
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = torch.utils.data.random_split(self, [n_train, n_val])
        return train_set, val_set


class MaskSplitByProfileDataset(MaskBaseDataset):
    """
        train / val 나누는 기준을 이미지에 대해서 random 이 아닌
        사람(profile)을 기준으로 나눕니다.
        구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다
        이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """
    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.choices(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {
            "train": train_indices,
            "val": val_indices
        }

    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = getattr(self.GenderLabels, gender)
                    age_label = self.AgeGroup.map_label(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self):
        return [Subset(self, indices) for phase, indices in self.indices.items()]

class MaskDataset(data.Dataset):
    num_classes = 3

    class MaskLabels:
        mask = 0
        incorrect = 1
        normal = 2

    _file_names = {
        "mask1": MaskLabels.mask,
        "mask2": MaskLabels.mask,
        "mask3": MaskLabels.mask,
        "mask4": MaskLabels.mask,
        "mask5": MaskLabels.mask,
        "incorrect_mask": MaskLabels.incorrect,
        "normal": MaskLabels.normal
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                # id, gender, race, age = profile.split("_")
                # gender_label = getattr(self.GenderLabels, gender)
                # age_label = self.AgeGroup.map_label(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                # self.gender_labels.append(gender_label)
                # self.age_labels.append(age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can takes huge amounts of time depending on your CPU machine :(")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        # gender_label = self.get_gender_label(index)
        # age_label = self.get_age_label(index)
        # multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, mask_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index):
        return self.mask_labels[index]

    # def get_gender_label(self, index):
    #     return self.gender_labels[index]

    # def get_age_label(self, index):
    #     return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    # @staticmethod
    # def encode_multi_class(mask_label, gender_label, age_label):
    #     return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label):
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8) # 값을 0에서 255로 제한
        return img_cp

    def split_dataset(self):
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = torch.utils.data.random_split(self, [n_train, n_val])
        return train_set, val_set

class GenderDataset(data.Dataset):
    num_classes = 2
    class MaskLabels:
        mask = 0
        incorrect = 1
        normal = 2
    class GenderLabels:
        male=0
        female=1
    _file_names = {
        "mask1": MaskLabels.mask,
        "mask2": MaskLabels.mask,
        "mask3": MaskLabels.mask,
        "mask4": MaskLabels.mask,
        "mask5": MaskLabels.mask,
        "incorrect_mask": MaskLabels.incorrect,
        "normal": MaskLabels.normal
    }
    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)

                id, gender, race, age = profile.split("_")
                gender_label = getattr(self.GenderLabels, gender)
                # age_label = self.AgeGroup.map_label(age)

                self.image_paths.append(img_path)
                # self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                # self.age_labels.append(age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can takes huge amounts of time depending on your CPU machine :(")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        image = self.read_image(index)
        gender_label = self.get_gender_label(index)

        image_transform = self.transform(image)
        return image_transform, gender_label

    def __len__(self):
        return len(self.image_paths)

    def get_gender_label(self, index):
        return self.gender_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)
    @staticmethod
    def decode_multi_class(multi_class_label):
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8) # 값을 0에서 255로 제한
        return img_cp
    def split_dataset(self):
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = torch.utils.data.random_split(self, [n_train, n_val])
        return train_set, val_set
class AgeDataset(data.Dataset):
    num_classes = 3

    class MaskLabels:
        mask = 0
        incorrect = 1
        normal = 2

    class AgeGroup:
        map_label = lambda x: 0 if int(x) < 30 else 1 if int(x) < 60 else 2

    _file_names = {
        "mask1": MaskLabels.mask,
        "mask2": MaskLabels.mask,
        "mask3": MaskLabels.mask,
        "mask4": MaskLabels.mask,
        "mask5": MaskLabels.mask,
        "incorrect_mask": MaskLabels.incorrect,
        "normal": MaskLabels.normal
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                # gender_label = getattr(self.GenderLabels, gender)
                # age_label = self.AgeGroup.map_label(age)
                age_label = age

                self.image_paths.append(img_path)
                # self.mask_labels.append(mask_label)
                # self.gender_labels.append(gender_label)
                self.age_labels.append(int(age_label))
            

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can takes huge amounts of time depending on your CPU machine :(")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        image = self.read_image(index)
        # mask_label = self.get_mask_label(index)
        # gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        # multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)
        # int(age_label)=age_label
        if age_label==60:
            age_label=80
        image_transform = self.transform(image)
        return image_transform, age_label
    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index):
        return self.mask_labels[index]

    def get_gender_label(self, index):
        return self.gender_labels[index]

    def get_age_label(self, index):
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def decode_multi_class(multi_class_label):
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8) # 값을 0에서 255로 제한
        return img_cp

    def split_dataset(self):
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = torch.utils.data.random_split(self, [n_train, n_val])
        return train_set, val_set
class AgeDataset_1(data.Dataset):

    def __init__(self, csv_file, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), transform=None,val_ratio=0.2):
        
        self.csv_file = csv_file
        self.root_dir = data_dir
        self.transform = None
        self.mean = mean
        self.std = std
        self.val_ratio=val_ratio
        self.setup()
    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir,
                                self.image_info[idx][0]+'_face.jpg')
        image = Image.open(img_path)
        age=self.image_info[idx][1]
        # print('zz')
        if self.transform:
            image_transformed = self.transform(image)

        return image_transformed,int(age)
    def setup(self):
        csv_data=pd.read_csv(self.csv_file)
        feature_data=csv_data[['file_name','real_age']]
        tonumpy=feature_data.to_numpy()
        self.image_info =np.vstack({tuple(row) for row in tonumpy})
    def set_transform(self, transform):
        self.transform = transform
    def split_dataset(self):
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = torch.utils.data.random_split(self, [n_train, n_val])
        return train_set, val_set
class MultiLabelDataset(data.Dataset):
    num_classes = 8

    class MaskLabels:
        mask = 0
        incorrect = 1
        normal = 2

    class GenderLabels:
        male = 3
        female = 4

    class AgeGroup:
        map_label = lambda x: '0' if int(x) < 30 else '1' if int(x) < 60 else '2'

    _file_names = {
        "mask1": 'mask',
        "mask2": 'mask',
        "mask3": 'mask',
        "mask4": 'mask',
        "mask5": 'mask',
        "incorrect_mask": 'incorrect',
        "normal": 'normal'
    }

    labels=[
        ("mask", "male",'0'),
        ("mask", "male",'1'),
        ("mask", "male",'2'),
        ("mask", "female",'0'),
        ("mask", "female",'1'),
        ("mask", "female",'2'),
        ("incorrect", "male",'0'),
        ("incorrect", "male",'1'),
        ("incorrect", "male",'2'),
        ("incorrect", "female",'0'),
        ("incorrect", "female",'1'),
        ("incorrect", "female",'2'),
        ("normal", "male",'0'),
        ("normal", "male",'1'),
        ("normal", "male",'2'),
        ("normal", "female",'0'),
        ("normal", "female",'1'),
        ("normal", "female",'2')
    ]
    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(self.labels)
    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = gender
                age_label = self.AgeGroup.map_label(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can takes huge amounts of time depending on your CPU machine :(")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        # multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)
       
        multi_label=self.mlb.transform([(mask_label, gender_label,age_label)])
        # print(multi_label[0]) 
        image_transform = self.transform(image)
        return image_transform, multi_label[0]

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index):
        return self.mask_labels[index]

    def get_gender_label(self, index):
        return self.gender_labels[index]

    def get_age_label(self, index):
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label):
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label):
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8) # 값을 0에서 255로 제한
        return img_cp

    def split_dataset(self):
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = torch.utils.data.random_split(self, [n_train, n_val])
        return train_set, val_set

class TestDataset(data.Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        # self.transform = transforms.Compose([
        #     Resize(resize, Image.BILINEAR),
        #     ToTensor(),
        #     Normalize(mean=mean, std=std),
        # ])
        if resize==0:
            self.transform = transforms.Compose([
            CenterCrop((380, 380)),
            # Resize(resize, Image.BILINEAR),
            # ColorJitter(0.1, 0.1, 0.1, 0.1),
            ToTensor(),
            Normalize(mean=mean, std=std),
            # AddGaussianNoise()
        ])
        else:
            self.transform = transforms.Compose([
                CenterCrop((380, 380)),
                Resize(resize, Image.BILINEAR),
                # ColorJitter(0.1, 0.1, 0.1, 0.1),
                ToTensor(),
                Normalize(mean=mean, std=std),
                # AddGaussianNoise()
            ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

class TestGrayDataset(data.Dataset):
    def __init__(self, img_paths, resize):
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            # Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

