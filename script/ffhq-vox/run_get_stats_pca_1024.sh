#!/usr/bin/env bash
python get_stats_pca.py --batchSize 4000 \
  --save_pca_path pca_stats/ffhq_1024 \
  --pca_iterations 250 \
  --latent_dimension 512 \
  --img_g_weights pretrained_models/stylegan2-ffhq-config-f.pt \
  --style_gan_size 1024 \
  --gpu 0

  
