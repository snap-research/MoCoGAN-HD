#!/usr/bin/env bash
python get_stats_pca.py --batchSize 4000 \
  --save_pca_path pca_stats/afhq \
  --pca_iterations 250 \
  --latent_dimension 512 \
  --img_g_weights pretrained_models/afhq-dog-fid7.8476-005655.pt \
  --style_gan_size 512 \
  --gpu 0

  
