_base_ = [
    '../_base_/models/upernet_beit.py', '../_base_/datasets/ade20k_640x640.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (640, 640)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='VisionTransformerOurs',
        arch='base',
        img_size=640,
        patch_size=16,
        out_indices=(3, 5, 7, 11),
        drop_path_rate=0.1,
        out_type='featmap',
        num_cls_tokens=1,
        init_cfg=dict(type='Pretrained', checkpoint='/path/to/ckpt', prefix='backbone.')
        ),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(426, 426)))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=3e-5, betas=(0.9, 0.999), weight_decay=0.05),
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=1)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
