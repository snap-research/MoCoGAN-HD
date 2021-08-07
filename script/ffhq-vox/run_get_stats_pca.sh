#!/usr/bin/env bash
python get_stats_pca.py --batchSize 4000 \
  --save_pca_path pca_stats/ffhq_256 \
  --pca_iterations 250 \
  --latent_dimension 512 \
  --img_g_weights pretrained_models/ffhq_256.pt \
  --style_gan_size 256 \
  --gpu 0

  
