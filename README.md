# TinyPredNet-A-Lightweight-Framework-for-Satellite-Image-Sequence-Prediction
Code for [TinyPredNet-A-Lightweight-Framework-for-Satellite-Image-Sequence-Prediction](https://dl.acm.org/doi/pdf/10.1145/3638773)
<img width="506" alt="image" src="https://github.com/bigfeetsmalltone/TinyPredNet-A-Lightweight-Framework-for-Satellite-Image-Sequence-Prediction/assets/30223491/4c9c5a57-a1dc-49ea-a810-21fef71abe8f">
<img width="495" alt="image" src="https://github.com/bigfeetsmalltone/TinyPredNet-A-Lightweight-Framework-for-Satellite-Image-Sequence-Prediction/assets/30223491/82a4287d-347b-4ba3-a766-923679a0b138">

## Abstract
Satellite image sequence prediction aims to precisely infer future satellite image frames with historical observations, which is a significant and challenging dense prediction task. Though existing deep learning models deliver promising performance for satellite image sequence prediction, the methods suffer from quite expensive training costs, especially in training time and GPU memory demand, due to the inefficiently modeling for temporal variations. This issue seriously limits the lightweight application in satellites such as space-borne forecast models. In this article, we propose a lightweight prediction framework TinyPredNet for satellite image sequence prediction, in which a spatial encoder and decoder model the intra-frame appearance features and a temporal translator captures inter-frame motion patterns. To efficiently model the temporal evolution of satellite image sequences, we carefully design a multi-scale temporal-cascaded structure and a channel attention-gated structure in the temporal translator. Comprehensive experiments are conducted on FengYun-4A (FY-4A) satellite dataset, which show that the proposed framework achieves very competitive performance with much lower computation cost compared to state-of-the-art methods. In addition, corresponding interpretability experiments are conducted to show how our designed structures work. We believe the proposed method can serve as a solid lightweight baseline for satellite image sequence prediction.

![image](./FengYun-4A.gif)

## FengYun Satellite Cloud Image Sequences Format
```
data path 
  ---seq 0 
      ---img 0.png 
      ---img 1.png
      ......
      ---img 23.png
  ---seq 1
  ---seq 2
```

## In / Out
Input: [1, 8, 1, 256, 256]
Ouput: [1, 16, 1, 256, 256]

## Training
```
python train.py --dataset 'satellite' --train_data_dir 'path of train dataset' \
--valid_data_dir 'path of val dataset' --checkpoint_save_dir './checkpoints' \
--img_size 256 --img_channel 1 --short_len 8 --long_len 16 \
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
```
## Test
```
python test.py \
--dataset 'satellite' --make_frame True \
--test_data_dir 'path of test dataset' --test_result_dir './test_outputs' \
--checkpoint_load_file 'model checkpoint' \
--img_size 256 --img_channel 1 \
--short_len 8 --out_len 16 \
--batch_size 1 \
--evaluate True \
--hid_S 64 \
--N_S 4 \
--in_channels 512 \
--out_channels 64 \
--reduced_dim 32 \
--scale 8 \
--expansion 8 \
--blocks 4
```
## Citation

If you are interested in our repository or our paper, please cite the following papers:

```
@article{dai2023tinyprednet,
  title={TinyPredNet: A Lightweight Framework for Satellite Image Sequence Prediction},
  author={Dai, Kuai and Li, Xutao and Lin, Huiwei and Jiang, Yin and Chen, Xunlai and Ye, Yunming and Xian, Di},
  journal={ACM Transactions on Multimedia Computing, Communications and Applications},
  year={2023},
  publisher={ACM New York, NY}
}

@article{dai2023learning,
  title={Learning Spatial-Temporal Consistency for Satellite Image Sequence Prediction},
  author={Dai, Kuai and Li, Xutao and Ma, Chi and Lu, Shenyuan and Ye, Yunming and Xian, Di and Tian, Lin and Qin, Danyu},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2023},
  publisher={IEEE}
}

@article{dai2022mstcgan,
  title={MSTCGAN: Multiscale time conditional generative adversarial network for long-term satellite image sequence prediction},
  author={Dai, Kuai and Li, Xutao and Ye, Yunming and Feng, Shanshan and Qin, Danyu and Ye, Rui},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--16},
  year={2022},
  publisher={IEEE}
}
```
