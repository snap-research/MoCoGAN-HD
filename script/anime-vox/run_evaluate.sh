python -W ignore evaluate.py \
  --save_pca_path pca_stats/anime \
  --latent_dimension 512 \
  --style_gan_size 512 \
  --img_g_weights pretrained_models/2020-01-11-skylion-stylegan2-animeportraits-networksnapshot-024664.pt \
  --load_pretrain_path checkpoints/anime \
  --load_pretrain_epoch 4 \
  --results results/anime \
  --num_test_videos 10 \
