python -W ignore evaluate.py  \
  --save_pca_path pca_stats/sky_timelapse \
  --latent_dimension 512 \
  --style_gan_size 128 \
  --img_g_weights pretrained_models/sky-fid10.8013-snapshot-012633.pt \
  --load_pretrain_path /path/to/checkpoints \
  --load_pretrain_epoch 0 \
  --results results/anime \
  --num_test_videos 10 \
