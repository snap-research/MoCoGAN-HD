python -W ignore evaluate.py  \
  --save_pca_path pca_stats/lsun_church \
  --latent_dimension 512 \
  --style_gan_size 256 \
  --img_g_weights pretrained_models/stylegan2-church-config-f.pt \
  --load_pretrain_path /path/to/checkpoints \
  --load_pretrain_epoch 0 \
  --results results/lsun_church \
  --num_test_videos 10 \
