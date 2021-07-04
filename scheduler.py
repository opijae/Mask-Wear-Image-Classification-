import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
_scheduler_dict={
        'step_lr' :StepLR,
        'Reduce_lr' : ReduceLROnPlateau,
        'cosine_lr' : CosineAnnealingLR
    }
def scheduler_entrypoint(scheduler_name):
    return _scheduler_dict[scheduler_name]

def create_scheduler(scheduler_name,optimizer,**kwargs):
    
    create_fn = scheduler_entrypoint(scheduler_name)
    if scheduler_name=='step_lr':
        scheduler = create_fn(optimizer, gamma=0.5)
    elif scheduler_name=='Reduce_lr':
        scheduler = create_fn(optimizer,factor=0.1, patience=10)
    elif scheduler_name=='cosine_lr':
        scheduler = create_fn(optimizer,T_max=2)
    return  scheduler


    # # -- scheduler: StepLR
    # # 지정된 step마다 learning rate를 감소시킵니다.
    # scheduler = StepLR(optimizer, lr_decay_step, gamma=0.5)

    # # -- scheduler: ReduceLROnPlateau
    # # 성능이 향상되지 않을 때 learning rate를 줄입니다. patience=10은 10회 동안 성능 향상이 없을 경우입니다.
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

    # # -- scheduler: CosineAnnealingLR
    # # CosineAnnealing은 learning rate를 cosine 그래프처럼 변화시킵니다.
    # scheduler = CosineAnnealingLR(optimizer, T_max=2, eta_min=0.)