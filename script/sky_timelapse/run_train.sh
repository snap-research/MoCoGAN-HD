python -W ignore train.py --name sky_timelapse \
  --time_step 2 \
  --lr 0.0001 \
  --save_pca_path pca_stats/sky_timelapse \
  --latent_dimension 512 \
  --dataroot /path/to/sky_timelapse \
  --checkpoints_dir checkpoints/sky_timelapse \
  --img_g_weights pretrained_models/sky-fid10.8013-snapshot-012633.pt \
  --multiprocessing_distributed --world_size 1 --rank 0 \
  --batchSize 32 \
  --workers 8 \
  --style_gan_size 128 \
  --total_epoch 200 \


  
