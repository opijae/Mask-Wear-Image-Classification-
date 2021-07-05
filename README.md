파일에 대한 설명
-------------

### age_inference.py
* 학습된 모델이 test데이터를 예측 할때 나이 부분이 잘되었는 확인

### dataset.py
* CustomAugmentation : 데이터 증강
* age_transform : 나이회귀 모델 때 사용한 데이터셋
* GrayAugmentation : gray scale 데이터셋 
* MaskBaseDataset : 데이터 로드 
* MaskSplitByProfileDataset : 한 사람이 train, valid에 따로 들어가지 않는 데이터셋 
* MaskDataset : 마스크 데이터셋 
* GenderDataset: 성별 데이터셋
* AgeDataset : 나이 데이터셋 (학습데이터에서 추출)
* AgeDataset_1 : 나이 데이터셋 (외부데이터에서 추출)
* MultiLabelDataset : multilabel classification에 대한 데이터셋
* TestDataset : 테스트 데이터셋
* TestGrayDataset : gray scale 테스트 데이터셋

### inference.py
* 학습한 모델을 가지고 csv파일 만들기

### loss.py
* loss functions

### model.py
* 여러 모델들 
    * 사전 학습 모델들
    * AgeModel : resnet34에 회귀를 붙임

### scheduler.py
* scheduler 모음집

### train.py
* 학습시 사용

### train_kfold.py
* kfold를 사용
* 한 에폭이 끝날때 마다 test데이터셋에 대한 추론 결과를 저장

### train_reg.py
* 나이회귀 학습을 위함

### vote.py
* 여러 모델들이 추론한 클래스 값(0~17)에 대한 투표를 진행해 새로운 csv 파일을 만듬

### vote_1.py
* 여러 모델들이 추론한 각 클래스에 대한 점수(18차원 벡터 값)에 대해 기하 평균을 사용해 csv파일 만듬

### test_dataset_eda.ipynb
* csv 값 visualize

### Report
* https://www.notion.so/Wrap-Up-e6630d63d62e4fc3847bedb723dd2e06
