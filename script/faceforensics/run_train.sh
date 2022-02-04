python -W ignore train.py --name faceforensics \
  --time_step 2 \
  --lr 0.0001 \
  --save_pca_path pca_stats/faceforensics \
  --latent_dimension 512 \
  --dataroot /path/to/faceforensics \
  --checkpoints_dir checkpoints/faceforensics \
  --img_g_weights pretrained_models/faceforensics-fid10.9920-snapshot-008765.pt \
  --multiprocessing_distributed --world_size 1 --rank 0 \
  --batchSize 16 \
  --workers 8 \
  --style_gan_size 256 \
  --total_epoch 100 \


  
