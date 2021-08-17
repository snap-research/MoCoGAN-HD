python -W ignore evaluate.py \
  --save_pca_path pca_stats/ffhq_1024 \
  --latent_dimension 512 \
  --style_gan_size 1024 \
  --img_g_weights pretrained_models/stylegan2-ffhq-config-f.pt \
  --load_pretrain_path /path/to/checkpoints \
  --load_pretrain_epoch 0 \
  --results results/ffhq_1024 \
  --num_test_videos 10 \
