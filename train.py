import argparse
import logging
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from dice_loss import dice_coeff

from eval import eval_net
#from unet.network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
#from unet.unet_model import UNet
from unet.unet_model import UNet
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

from copy import  deepcopy


def acquire_all_img(temp_path,temp_list):
    full_img_path=[]
    full_mask_path=[]

    for patient in temp_list:

        mask_path=os.path.join(temp_path,patient,'mask')
        all_mask=os.listdir(mask_path)
        all_mask=[os.path.join(mask_path,i) for i in all_mask]
        all_img=[i.replace('mask','img') for i in all_mask]

        full_img_path.extend(all_img)
        full_mask_path.extend(all_mask)

    return full_img_path,full_mask_path

def train_net(base_dir,net,
              device,
              epochs=15,
              batch_size=16,
              lr=0.001,
              val_percent=0.05,
              save_cp=True,
              img_scale=0.5):


    if base_dir=='../CAMUS':
        dataset_type='CAMUS'
        img_scale=1.0
    if base_dir=='../DYNAMIC':
        dataset_type='DYNAMIC'
        img_scale=1.0
    train_path=os.path.join(base_dir,'training')
    test_path=os.path.join(base_dir,'testing')

    train_patients=os.listdir(train_path)
    test_patients=os.listdir(test_path)

    train_img, train_mask = acquire_all_img(train_path, train_patients)
    test_img,test_mask=acquire_all_img(test_path,test_patients)


    train_dataset = BasicDataset(train_img,train_mask,scale=img_scale)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=24, pin_memory=True,drop_last=False)


    test_dataset = BasicDataset(test_img,test_mask,scale=img_scale)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=24, pin_memory=True,drop_last=False)


    n_train, n_val = len(train_dataset), len(test_dataset)

    global_step = 0
    best_score = 0
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {len(train_patients),len(train_dataset)}
        Validation size: {len(test_patients),len(test_dataset)}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    #optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8,amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2,patience=10,verbose=True,min_lr=1e-7,cooldown=20)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                masks_pred = net(imgs)

                loss = criterion(masks_pred, true_masks)
                loss_dsc = 1- dice_coeff(torch.sigmoid(masks_pred), true_masks).item()
                loss +=loss_dsc
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                




        if save_cp:

            scheduler.step(epoch_loss)
            print('Time :{},epoch:{},Net:{},LR:{},Validation_score:{}'.format(
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), epoch, net.name,
                str(float(optimizer.param_groups[0]['lr'])), str(str(val_score_mask))))

            model_path=os.path.join(dataset_type,dir_checkpoint,net.name)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(net.state_dict(),
                       os.path.join(model_path,f'CP_epoch{epoch + 1}.pth') )
            logging.info(f'Checkpoint {epoch + 1} saved !')
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=30,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=8,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1.0,
                        help='Downscaling factor of the images')
    parser.add_argument('-c','--checkpoint_path',type=str,default='checkpoints/')
    parser.add_argument('-d','--dataset_type',type=str,default='../CAMUS')
    parser.add_argument('--device_number',type=str,default='0')
    parser.add_argument('--select_strategy', type=str, default='mean')


    return parser.parse_args()


if __name__ == '__main__':
    torch.backends.cudnn.enabled =False
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_number
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    start_time=time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))
    dir_checkpoint = start_time
    model_type = [UNet(3,1,select_strategy=args.select_strategy)]
    for net in model_type:
        logging.info(f'Network:\t{str(net.name)}\n'
                     f'\t{net.n_channels} input channels\n'
                     f'\t{net.n_classes} output channels (classes)\n')

        if args.load:
            net.load_state_dict(
                torch.load(args.load, map_location=device)
            )
            logging.info(f'Model loaded from {args.load}')

        net.to(device=device)
        # faster convolutions, but more memory
        # cudnn.benchmark = True

        try:
            train_net(base_dir=args.dataset_type,
                      net=net,
                      epochs=args.epochs,
                      batch_size=args.batchsize,
                      lr=args.lr,
                      device=device,
                      img_scale=args.scale
                        )
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
