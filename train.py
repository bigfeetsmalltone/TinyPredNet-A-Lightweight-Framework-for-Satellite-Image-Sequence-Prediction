from model import Predictor
from dataloader import Radar, Satellite
from utils import *
# import torchsummary
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from fvcore.nn import FlopCountAnalysis

import lpips
import argparse
import numpy as np
import time
import os

from torch.utils.tensorboard import SummaryWriter

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='satellite',
                    help='training dataset (satellite or radar)')
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers')
parser.add_argument('--train_data_dir', type=str, default='enter_the_path',
                    help='directory of training set')
parser.add_argument('--valid_data_dir', type=str, default='enter_the_path',
                    help='directory of validation set')
parser.add_argument('--checkpoint_load', type=bool, default=False,
                    help='whether to load checkpoint')
parser.add_argument('--checkpoint_load_file', type=str, default='enter_the_path',
                    help='file path for loading checkpoint')
parser.add_argument('--checkpoint_save_dir', type=str, default='./checkpoints',
                    help='directory for saving checkpoints')

parser.add_argument('--log_path', type=str, default='./tensorboards',
                    help='directory for saving logs')

parser.add_argument('--img_size', type=int, default=64,
                    help='height and width of video frame')
parser.add_argument('--img_channel', type=int, default=1,
                    help='channel of video frame')
parser.add_argument('--short_len', type=int, default=10,
                    help='number of input short-term frames')
parser.add_argument('--out_len', type=int, default=30,
                    help='number of output predicted frames')

parser.add_argument('--batch_size', type=int, default=128,
                    help='mini-batch size')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate')
parser.add_argument('--iterations', type=int, default=300000,
                    help='number of total iterations')
parser.add_argument('--iterations_warmup', type=int, default=5,
                    help='number of iterations for warming up model')
parser.add_argument('--print_freq', type=int, default=100,
                    help='frequency of printing logs')

parser.add_argument('--in_channels', default=512, type=int)
parser.add_argument('--out_channels', default=64, type=int)
parser.add_argument('--reduced_dim', default=32, type=int)
parser.add_argument('--scale', default=8, type=int)
parser.add_argument('--expansion', default=8, type=int)
parser.add_argument('--blocks', default=4, type=int)
  
parser.add_argument('--hid_S', default=64, type=int)
parser.add_argument('--N_S', default=4, type=int)

args = parser.parse_args()


if __name__ == '__main__':
    if not os.path.isdir(args.checkpoint_save_dir):
        os.makedirs(args.checkpoint_save_dir)

    # deine the tensorboard log
    writer = SummaryWriter(args.log_path)

    # define the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred_model = Predictor(args).to(device)
    pred_model = nn.DataParallel(pred_model)

    # FLOPs and parameters
    input = (torch.randn((1, 8, 1, 256, 256)).cuda())
    flops = FlopCountAnalysis(pred_model, input)
    print(f'compuatation FLOPs: {flops.total() / 1e9}G')

    total = sum([param.nelement() for param in pred_model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
    # torchsummary.summary(pred_model, [(5, 1, 128, 128), (25, 1, 128, 128),(5, 1, 128, 128), (25, 1, 128, 128)])
    # optionally load checkpoint
    if args.checkpoint_load:
        pred_model.load_state_dict(torch.load(args.checkpoint_load_file))
        print('Checkpoint is loaded from ' + args.checkpoint_load_file)
    
    # prepare dataloader for selected dataset
    if args.dataset == 'radar':
        train_dataset = Radar_train(args.train_data_dir, seq_len=args.short_len+args.out_len, train=True)
        print(len(train_dataset))
        trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
        valid_dataset = Radar(args.valid_data_dir, seq_len=args.short_len+args.out_len, train=False)
        print(len(valid_dataset))
        validloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)
    elif args.dataset == 'satellite':
        train_dataset = Satellite(args.train_data_dir, seq_len=args.short_len+args.out_len, train=True)
        print(len(train_dataset))
        trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
        valid_dataset = Satellite(args.valid_data_dir, seq_len=args.short_len+args.out_len, train=False)
        print(len(valid_dataset))
        validloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)
    
    # define optimizer and loss function
    optimizer = torch.optim.Adam(pred_model.parameters(), lr=args.lr)
    l1_loss, l2_loss = nn.L1Loss().to(device), nn.MSELoss().to(device)
    lpips_dist = lpips.LPIPS(net = 'alex').to(device)

    mse_min, psnr_max, ssim_max, lpips_min = 99999, 0, 0, 99999
    train_loss = AverageMeter()
    valid_mse, valid_psnr, valid_ssim, valid_lpips = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    print('Start training...')
    
    start_time = time.time()
    data_iterator = iter(trainloader)
    for train_i in range(args.iterations):
        try:
            train_data = next(data_iterator)
        except:
            data_iterator = iter(trainloader)
            train_data = next(data_iterator)

        # define data indexes
        short_start, short_end = 0, args.short_len
        out_gt_start, out_gt_end = short_end, short_end+args.out_len
        # print(long_start)
        # obtain input data and output gt
        train_data = torch.stack(train_data).to(device)
        train_data = train_data.transpose(dim0=0, dim1=1) # make (N, T, C, H, W)
        short_data = train_data[:, short_start:short_end, :, :, :]
        out_gt = train_data[:, out_gt_start:out_gt_end, :, :, :]

        pred_model.train()

        out_pred = pred_model(short_data)
        loss_p1 = l1_loss(out_pred, out_gt) + l2_loss(out_pred, out_gt)
        optimizer.zero_grad()
        loss_p1.backward()
        optimizer.step()

        writer.add_scalar(f'loss-basic', loss_p1, train_i)
        
        train_loss.update(float(loss_p1))

        if (train_i+1) % args.print_freq == 0:
            # preserve the latest 10 models
            torch.save(pred_model.state_dict(), args.checkpoint_save_dir+'/trained_file_'+str(((train_i+1) // args.print_freq)%10)+'.pt')

            elapsed_time = time.time() - start_time; start_time = time.time()
            print('elapsed time: {:.0f} sec'.format(elapsed_time))
            
