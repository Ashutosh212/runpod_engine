# Dataset config for 8-class fine-tuning (merged 3-source dataset)
# Classes: arty, camo, logistic, missile, radar, smallvehicle, tank, vehicle
# Sources merged into /sfs/merged_train_all/:
#   /sfs/data/custom/trainval   — 5541 images (absolute DOTA coords)
#   /sfs/syn_all                — 13189 images (coords converted from normalised; class names remapped)
#   /sfs/up42_train_2ndstage_may_1/patches — 8836 images (absolute DOTA coords)
# Total: 27566 images
# Annotation counts (train+val merged):
#   arty:1407  camo:4989  logistic:13536  missile:14012  radar:10550
#   smallvehicle:7186  tank:7607  vehicle:31897
# Class weights: inverse-sqrt-frequency, normalised so average = 1.0
#   [2.21, 1.17, 0.71, 0.70, 0.81, 0.98, 0.95, 0.46]
# Val set: patches only (/sfs/up42_train_2ndstage_may_1/patches/)
dataset_type = 'DOTADataset'
data_root = '/sfs/merged_train_all/'
val_data_root = '/sfs/up42_test_qc_may_w1_2/patches/'
backend_args = None

metainfo = dict(
    classes=('arty', 'camo', 'logistic', 'missile', 'radar',
             'smallvehicle', 'tank', 'vehicle'),
    palette=[(220, 20, 60),    # arty         — crimson red
             (85, 107, 47),    # camo         — dark olive green
             (255, 140, 0),    # logistic     — dark orange
             (138, 43, 226),   # missile      — blue violet
             (0, 206, 209),    # radar        — dark turquoise
             (135, 206, 235),  # smallvehicle — sky blue
             (34, 139, 34),    # tank         — forest green
             (255, 215, 0)])   # vehicle      — gold

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='mmdet.RandomResize',
         scale=(1280, 720),
         ratio_range=(0.5, 1.5),
         keep_ratio=True),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.RandomFlip',
         prob=0.75,
         direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='RandomRotate',
         prob=0.5,
         angle_range=180,
         rect_obj_labels=[]),
    dict(type='mmdet.PackDetInputs')
]

val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(1280, 720), keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annfiles/',
        data_prefix=dict(img_path='images/'),
        filter_cfg=dict(filter_empty_gt=True),
        img_suffix='png',
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=val_data_root,
        metainfo=metainfo,
        ann_file='labels/',
        data_prefix=dict(img_path='images/'),
        test_mode=True,
        img_suffix='png',
        pipeline=val_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='DOTAMetric', metric='mAP')
test_evaluator = val_evaluator
