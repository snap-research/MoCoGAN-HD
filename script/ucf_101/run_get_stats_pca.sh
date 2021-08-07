#!/usr/bin/env bash
python get_stats_pca.py --batchSize 4000 \
  --save_pca_path pca_stats/ucf_101 \
  --pca_iterations 250 \
  --latent_dimension 512 \
  --img_g_weights pretrained_models/ucf-256-fid41.6761-snapshot-006935.pt \
  --style_gan_size 256 \
  --gpu 0

  
