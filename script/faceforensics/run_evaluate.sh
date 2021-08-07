python -W ignore evaluate.py \
  --save_pca_path pca_stats/faceforensics \
  --latent_dimension 512 \
  --style_gan_size 256 \
  --img_g_weights pretrained_models/faceforensics-fid10.9920-snapshot-008765.pt \
  --load_pretrain_path checkpoints/faceforensics \
  --load_pretrain_epoch 749 \
  --results results/faceforensics \
  --num_test_videos 10 \
