#!/usr/bin/env bash
python get_stats_pca.py --batchSize 4000 \
  --save_pca_path pca_stats/lsun_church \
  --pca_iterations 250 \
  --latent_dimension 512 \
  --img_g_weights pretrained_models/stylegan2-church-config-f.pt \
  --style_gan_size 256 \
  --gpu 0

  
