from .include.detrpose_hgnetv2 import model, criterion, training_params, postprocessor
from .include.dataset import dataset_train, dataset_val, dataset_test, evaluator

from src.core import LazyCall as L
from src.nn.optimizer import ModelEMA 
from src.misc.get_param_dicts import get_optim_params

from torch import optim

training_params.output_dir =  "output/detrpose_hgnetv2_l"
training_params.epochs = 52 # 48 + 4
training_params.use_ema = True

ema = L(ModelEMA)(
    decay=0.9999,
    warmups=2000
    )

# optimizer params
optimizer = L(optim.AdamW)(
    params=L(get_optim_params)(
        cfg=[
                {
                'params': '^(?=.*backbone).*$',
                'lr': 0.00001
                },
            ],
        # model=model
        ),
    lr=0.0001,
    betas=[0.9, 0.999],
    weight_decay=0.0001
    )

lr_scheduler = L(optim.lr_scheduler.MultiStepLR)(
    # optimizer=optimizer,
    milestones=[1000],
    gamma=0.1
    )

