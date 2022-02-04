python -W ignore train.py --name lsun_church-tlvdb \
  --time_step 2 \
  --lr 0.0001 \
  --save_pca_path pca_stats/lsun_church \
  --latent_dimension 512 \
  --dataroot /path/to/tlvdb \
  --checkpoints_dir checkpoints/lsun \
  --img_g_weights pretrained_models/stylegan2-church-config-f.pt \
  --multiprocessing_distributed --world_size 1 --rank 0 \
  --batchSize 16 \
  --workers 8 \
  --style_gan_size 256 \
  --total_epoch 200 \
  --cross_domain \


  
