python -W ignore train.py --name ffhq_256-voxel \
  --time_step 2 \
  --lr 0.0001 \
  --save_pca_path pca_stats/ffhq_256 \
  --latent_dimension 512 \
  --dataroot /path/to/voxel \
  --checkpoints_dir checkpoints/ffhq \
  --img_g_weights pretrained_models/ffhq_256.pt \
  --multiprocessing_distributed --world_size 1 --rank 0 \
  --batchSize 16 \
  --workers 8 \
  --w_match 5.0 \
  --style_gan_size 256 \
  --total_epoch 30 \
  --cross_domain \


  
