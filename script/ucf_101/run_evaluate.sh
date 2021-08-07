python -W ignore evaluate.py  \
  --save_pca_path pca_stats/ucf_101 \
  --latent_dimension 512 \
  --style_gan_size 256 \
  --img_g_weights pretrained_models/ucf-256-fid41.6761-snapshot-006935.pt \
  --load_pretrain_path /path/to/checkpoints \
  --load_pretrain_epoch 0 \
  --results results/ucf_101 \
  --num_test_videos 10 \
