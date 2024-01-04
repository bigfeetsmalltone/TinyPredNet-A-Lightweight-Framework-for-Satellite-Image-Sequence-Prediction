python train.py --dataset 'satellite' --train_data_dir 'path of train dataset' \
--valid_data_dir 'path of val dataset' --checkpoint_save_dir './checkpoints' \
--img_size 256 --img_channel 1 --short_len 8 \
--out_len 16 --batch_size 8 --lr 0.0002 \
--iterations 100000 --print_freq 1000 \
--hid_S 64 \
--N_S 4 \
--in_channels 512 \
--out_channels 64 \
--reduced_dim 32 \
--scale 8 \
--expansion 8 \
--blocks 4