python -W ignore train.py --name ffhq_1024-voxel \
  --time_step 2 \
  --lr 0.0001 \
  --save_pca_path pca_stats/ffhq_1024 \
  --latent_dimension 512 \
  --dataroot /path/to/voxel \
  --checkpoints_dir checkpoints/ffhq_1024 \
  --img_g_weights pretrained_models/stylegan2-ffhq-config-f.pt \
  --multiprocessing_distributed --world_size 1 --rank 0 \
  --batchSize 16 \
  --workers 8 \
  --w_match 5.0 \
  --style_gan_size 1024 \
  --total_epoch 30 \
  --cross_domain \


  
