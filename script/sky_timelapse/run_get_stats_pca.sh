    #!/usr/bin/env bash
    python get_stats_pca.py --batchSize 4000 \
    --save_pca_path pca_stats/sky_timelapse \
    --pca_iterations 250 \
    --latent_dimension 512 \
    --img_g_weights pretrained_models/network-snapshot-002880-best.pkl \
    --style_gan_size 256 \
    --gpu 0 
    
