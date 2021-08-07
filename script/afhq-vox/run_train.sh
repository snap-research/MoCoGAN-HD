python -W ignore train.py --name afhq-voxel \
  --time_step 2 \
  --lr 0.0001 \
  --save_pca_path pca_stats/afhq \
  --latent_dimension 512 \
  --dataroot /path/to/voxel \
  --checkpoints_dir checkpoints/afhq \
  --img_g_weights pretrained_models/afhq-dog-fid7.8476-005655.pt \
  --multiprocessing_distributed --world_size 1 --rank 0 \
  --batchSize 8 \
  --workers 8 \
  --style_gan_size 512 \
  --total_epoch 100 \
  --cross_domain \


  
