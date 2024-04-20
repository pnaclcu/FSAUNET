import argparse
import logging
import os
from tqdm import  tqdm
import numpy as np
import torch

from PIL import Image
from utils.dataset import BasicDataset
from unet.unet_model import UNet
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import cv2


def acquire_all_img(temp_path, temp_list):
    full_img_path = []
    full_mask_path = []

    for patient in temp_list:
        mask_path = os.path.join(temp_path, patient, 'mask')
        all_mask = os.listdir(mask_path)
        all_mask = [os.path.join(mask_path, i) for i in all_mask]
        all_img = [i.replace('mask', 'img') for i in all_mask]

        full_img_path.extend(all_img)
        full_mask_path.extend(all_mask)

    return full_img_path, full_mask_path
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='2023-07-27 04:00:05/',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',default='./img_hight_resolution',
                        help='filenames of input images')

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)


    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []


    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    from torch.utils.data import DataLoader, random_split
    args = get_args()

    from skimage import img_as_ubyte
    from dice_loss import dice_coeff
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_dir='../CAMUS'
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



    test_dataset = BasicDataset(test_img,test_mask,scale=img_scale)
    val_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=24, pin_memory=True,drop_last=False)

    # n_train, n_val = len(train_dataset), len(test_dataset)
    out_files = get_output_filenames(args)

    model_type = [UNet(n_channels=3, n_classes=1)]
    net=model_type[0]

    ckpts_times_path=base_dir[1:]
    all_times=os.listdir(ckpts_times_path)
    for time in all_times:
        time_path=os.path.join(ckpts_times_path,time)
        ckpt_dir_path=os.path.join(time_path,'FSAUNet')
        all_ckpts=os.listdir(ckpt_dir_path)

        for ckpt in all_ckpts:

            ckpt_path=os.path.join(ckpt_dir_path,ckpt)
            net.load_state_dict(torch.load(ckpt_path, map_location=device))
            net.to(device)
            net.eval()

            for batch in tqdm(val_loader):
                tot=0

                imgs, true_masks ,img_name= batch['image'], batch['mask'],batch['name']
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device)

                with torch.no_grad():
                    mask_preds = net(imgs)
                    mask_preds = torch.sigmoid(mask_preds)
                    mask_preds = (mask_preds > 0.5).float()

                for i in range(mask_preds.shape[0]):
                    dice_mask=dice_coeff(mask_preds[i],true_masks[i]).item()
                    current_img_name=img_name[i].split('/')[-1].split('.')[0]
                    current_patient_name=img_name[i].split('/')[3]
                    if base_dir == '../CAMUS':
                        abs_path = ckpt_path.replace('CAMUS', 'CAMUS_perdiction')
                        abs_path_mask=os.path.join(abs_path,'mask')


                    if base_dir == '../DYNAMIC':
                        abs_path = ckpt_path.replace('DYNAMIC', 'DYNAMIC_perdiction')
                        abs_path_mask=os.path.join(abs_path,'mask')

                    if not os.path.exists(abs_path_mask):
                        os.makedirs(abs_path_mask)


                    if base_dir == '../CAMUS':
                        mask_final_path=os.path.join(abs_path_mask,current_img_name+'_'+str(dice_mask)+'.png')
                        cv2.imwrite(mask_final_path,
                                  np.array(img_as_ubyte(mask_preds[i, :,:, :].permute(1,2,0).to('cpu') > 0.5)))

                    if base_dir =='../DYNAMIC':
                        mask_final_path=os.path.join(abs_path_mask,current_patient_name+'_'+current_img_name+'_'+str(dice_mask)+'.png')
                        cv2.imwrite(mask_final_path,
                                  np.array(img_as_ubyte(mask_preds[i, :,:, :].permute(1,2,0).to('cpu') > 0.5)))

