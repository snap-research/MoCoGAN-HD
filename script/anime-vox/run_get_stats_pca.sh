#!/usr/bin/env bash
python get_stats_pca.py --batchSize 4000 \
  --save_pca_path pca_stats/anime \
  --pca_iterations 250 \
  --latent_dimension 512 \
  --img_g_weights pretrained_models/2020-01-11-skylion-stylegan2-animeportraits-networksnapshot-024664.pt \
  --style_gan_size 512 \
  --gpu 0

  
