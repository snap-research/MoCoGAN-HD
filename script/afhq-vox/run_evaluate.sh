python -W ignore evaluate.py \
  --save_pca_path pca_stats/afhq \
  --latent_dimension 512 \
  --style_gan_size 512 \
  --img_g_weights pretrained_models/afhq-dog-fid7.8476-005655.pt \
  --load_pretrain_path checkpoints/afhq \
  --load_pretrain_epoch 7 \
  --results results/afhq \
  --num_test_videos 10 \
