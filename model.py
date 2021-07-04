import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision
class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class efficientnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        # from torchvision.models import eff
        
        self.net=timm.create_model('efficientnet_b0', pretrained=True)
        self.net.classifier=nn.Sequential(
            # nn.Linear(1280,1280),
            # nn.Dropout(0.5),
            nn.Linear(1280,num_classes)
        )

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x=self.net(x)
        return x

class efficientnet_b4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        from efficientnet_pytorch import EfficientNet
        self.net = EfficientNet.from_pretrained('efficientnet-b4')
        self.net.classifier=nn.Sequential(
            nn.Linear(1792,num_classes)
        )

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x=self.net(x)
        return x

# Custom Model Template
class efficientnet_gray(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        # from torchvision.models import eff
        
        self.net=timm.create_model('efficientnet_b0', pretrained=True)
        self.net.conv_stem=nn.Conv2d(1,32,kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False)
        self.net.classifier=nn.Sequential(
            nn.Linear(1280,num_classes)
        )

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x=self.net(x)
        return x       
class efficientnet_multilabel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        # from torchvision.models import eff
        
        self.net=timm.create_model('efficientnet_b0', pretrained=True)
        # self.net.conv_stem=nn.Conv2d(1,32,kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False)
        self.net.classifier=nn.Sequential(
            nn.Linear(1280,num_classes)
        )
        self.sig=nn.Sigmoid()

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x=self.sig(self.net(x))
        return x      
# Custom Model Template
class xception_multilabel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        # from torchvision.models import eff
        
        self.net=timm.create_model('xception', pretrained=True)
        self.net.fc=nn.Sequential(
            nn.Linear(2048,num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x=self.net(x)
        return x
# Custom Model Template
class xception(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        # from torchvision.models import eff
        
        self.net=timm.create_model('xception', pretrained=True)
        self.net.fc=nn.Sequential(
            nn.Linear(2048,num_classes)
            # nn.Dropout(0.5),
            # nn.Linear(1024,512),
            # nn.Dropout(0.5),
            # nn.Linear(512,num_classes)
        )

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x=self.net(x)
        return x
# Custom Model Template
class resnext50_32x4d(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        # from torchvision.models import eff
        
        self.net=timm.create_model('resnext50_32x4d', pretrained=True)
        self.net.fc=nn.Sequential(
            # nn.Linear(2048,1024),
            # nn.Dropout(0.5),
            # nn.Linear(1024,512),
            # nn.Dropout(0.5),
            # nn.Linear(512,num_classes)
            nn.Linear(2048,num_classes)
        )

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x=self.net(x)
        return x

class AgeModel(nn.Module):
    def __init__(self):
        super().__init__()
        layers = list(torchvision.models.resnet34(pretrained=True).children())[:-2]
        layers += [nn.AdaptiveAvgPool2d(1)]
        layers += [nn.AdaptiveMaxPool2d(output_size=1)]
        layers += [nn.Flatten()]
        layers += [nn.Linear(512, 256, bias=True), nn.ReLU(inplace=True)]
        layers += [nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
        layers += [nn.Dropout(p=0.50)]
        layers += [nn.Linear(256, 16, bias=True), nn.ReLU(inplace=True)]
        layers += [nn.Linear(16,1)]
        self.agemodel = nn.Sequential(*layers)
    def forward(self, x):
        return self.agemodel(x).squeeze(-1)
