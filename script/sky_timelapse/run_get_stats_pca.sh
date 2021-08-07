#!/usr/bin/env bash
python get_stats_pca.py --batchSize 4000 \
  --save_pca_path pca_stats/sky_timelapse \
  --pca_iterations 250 \
  --latent_dimension 512 \
  --img_g_weights pretrained_models/sky-fid10.8013-snapshot-012633.pt \
  --style_gan_size 128 \
  --gpu 0

  
