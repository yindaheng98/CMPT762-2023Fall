from encoder_decoder import EncoderDecoder


def segnext():
    return EncoderDecoder(
        backbone=dict(
            embed_dims=[32, 64, 160, 256],
            mlp_ratios=[8, 8, 4, 4],
            drop_rate=0.0,
            drop_path_rate=0.1,
            depths=[3, 3, 5, 2],
            norm_cfg=dict(type='SyncBN', requires_grad=True)),
        decode_head=dict(
            in_channels=[64, 160, 256],
            in_index=[1, 2, 3],
            channels=256,
            ham_channels=256,
            ham_kwargs=dict(MD_R=16),
            dropout_ratio=0.1,
            num_classes=19,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    )

if __name__ == "__main__":
    print(segnext())