from src.core import LazyCall as L
from src.models.detrpose import (
    DETRPose,
    HybridEncoder,
    Transformer,
    PostProcess,
    Criterion,
    HungarianMatcher,
    )

from src.nn import HGNetv2

training_params = {
    "clip_max_norm": 0.1,
    "save_checkpoint_interval": 1,
    "grad_accum_steps": 1,
    "print_freq": 100,
    'sync_bn': True,
    'use_ema': False,
    'dist_url': 'env://',
}

eval_spatial_size = (640, 640)
hidden_dim = 256
n_levels = 3
feat_strides = [8, 16, 32]
num_classes = 2

model = L(DETRPose)(
    backbone=L(HGNetv2)(
        name='B4',
        use_lab=False,
        return_idx=[1, 2, 3],
        freeze_stem_only=True,
        freeze_at=-1,
        freeze_norm=True,
        pretrained=True,
        ),
	encoder=L(HybridEncoder)(
        in_channels=[512, 1024, 2048],
        feat_strides=feat_strides,
        n_levels=n_levels,
        hidden_dim=hidden_dim,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act='gelu',
        expansion=1.0,
        depth_mult=1.0,
        act='silu',
        temperatureH=20,
        temperatureW=20,
        eval_spatial_size= eval_spatial_size
		),
	transformer=L(Transformer)(
        hidden_dim=hidden_dim,
        dropout=0.0,
        nhead=8,
        num_queries=60,
        dim_feedforward=1024,
        num_decoder_layers=6,
        normalize_before=False,
        return_intermediate_dec=True,
        activation='relu',
        num_feature_levels=3,
        dec_n_points=4,
        learnable_tgt_init=True,
        two_stage_type='standard',
        num_body_points=17,
        aux_loss=True,
        num_classes=num_classes,
        dec_pred_class_embed_share = False,
        dec_pred_pose_embed_share = False,
        two_stage_class_embed_share=False,
        two_stage_bbox_embed_share=False,
        cls_no_bias = False,
        # new parameters
        feat_strides=[8, 16, 32],
        eval_spatial_size=eval_spatial_size,
        reg_max=32,
        reg_scale=4
        ),
    )

criterion = L(Criterion)(
	num_classes=num_classes,
	weight_dict={'loss_vfl': 2.0, 'loss_keypoints': 10.0, 'loss_oks': 4.0}, 
	focal_alpha=0.25,
	losses=['vfl', 'keypoints'], 
	matcher=L(HungarianMatcher)(
		cost_class=2.0,
		cost_keypoints=10.0,
        cost_oks=4.0,
		focal_alpha=0.25
		),
    num_body_points=17
	)

postprocessor = L(PostProcess)(num_select=60, num_body_points=17)
