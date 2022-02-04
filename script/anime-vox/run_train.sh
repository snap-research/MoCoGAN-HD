python -W ignore train.py --name anime-voxel \
  --time_step 2 \
  --lr 0.0001 \
  --save_pca_path pca_stats/anime \
  --latent_dimension 512 \
  --dataroot /path/to/voxel \
  --checkpoints_dir checkpoints/anime \
  --img_g_weights pretrained_models/2020-01-11-skylion-stylegan2-animeportraits-networksnapshot-024664.pt \
  --multiprocessing_distributed --world_size 1 --rank 0 \
  --batchSize 8 \
  --workers 8 \
  --style_gan_size 512 \
  --total_epoch 100 \
  --cross_domain \


  
