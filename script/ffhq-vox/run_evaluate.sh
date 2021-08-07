python -W ignore evaluate.py \
  --save_pca_path pca_stats/ffhq_256 \
  --latent_dimension 512 \
  --style_gan_size 256 \
  --img_g_weights pretrained_models/ffhq_256.pt \
  --load_pretrain_path /path/to/checkpoints \
  --load_pretrain_epoch 0 \
  --results results/ffhq_256 \
  --num_test_videos 10 \
