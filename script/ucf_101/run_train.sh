python -W ignore train.py --name ucf_101 \
  --time_step 2 \
  --lr 0.0001 \
  --save_pca_path pca_stats/ucf_101 \
  --latent_dimension 512 \
  --dataroot /path/to/ucf_101 \
  --checkpoints_dir checkpoints/ucf \
  --img_g_weights pretrained_models/ucf-256-fid41.6761-snapshot-006935.pt \
  --multiprocessing_distributed --world_size 1 --rank 0 \
  --batchSize 16 \
  --workers 8 \
  --style_gan_size 256 \
  --total_epoch 20 \


  
