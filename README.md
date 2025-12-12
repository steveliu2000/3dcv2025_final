python train.py configs/config_train.yaml resume=pretrained_models/SMIRK_em1.pt train.loss_weights.emotion_loss=1.0 train.use_temporal_loss=true train.log_path=logs/temporal

python train.py configs/config_train.yaml resume=pretrained_models/SMIRK_em1.pt train.loss_weights.emotion_loss=1.0 train.use_temporal_loss=false train.log_path=logs/non_temporal