import os
import cv2
import time
import torch
import argparse
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from unet import UNet as my_model
from dataset import TestDataset
from utils import init_distributed_mode
import segmentation_models_pytorch as smp

#test function
def Qualitative(test_loader, model, save_path):
    save_path = os.path.join(save_path, 'figures')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(test_loader.size)):
            image, target, name  = test_loader.load_data()
            clip_name = name.split("/")[-2] # e.g. '1'
            img_name  = name.split("/")[-1] # e.g. '001.png'
            image     = image.cuda()
            pred,_ ,_   = model(image)
            pred      = F.interpolate(pred, size=target.shape, mode='bilinear', align_corners=False)

            pred       = (pred[0,0] > 0)
            pred       = pred.cpu().numpy()
            pred       = np.expand_dims(pred,2)

            # save
            save_video_path = os.path.join(save_path, clip_name)
            if not os.path.exists(save_video_path):
                os.makedirs(save_video_path, exist_ok=True)
            cv2.imwrite(os.path.join(save_video_path, img_name), np.uint8((pred)*255))
    return
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone',    type=str,   default='res2net50',               help='model backbone'                       )
    parser.add_argument('--pretrained',  type=str,   default=None,                      help='pretrained model path'                )
    parser.add_argument('--resume',      type=str,   default='./results/unet/log_2024-12-20_13:34:12',                      help='resume path')
    parser.add_argument('--clip_size',   type=int,   default=3,                         help='a clip size'                          )
    parser.add_argument('--train_size',  type=int,   default=352,                       help='training dataset size'                )
    parser.add_argument('--gpu_id',      type=str,   default='0',                       help='train use gpu'                        )
    parser.add_argument('--data_root',   type=str,   default='./data',                      help='the training images root'             )
    parser.add_argument('--task',        type=str,   default='or',                      help='Quantitative or Qualitative eval task')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    cudnn.benchmark = True

    model = my_model()
    model.cuda()

    if args.resume:
        print('loading model...')
        load_path = args.resume + '/epoch_bestDice.pth'
        checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
        model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint.items()})
        print('load model from ', load_path)

    print('load data...')
    test_loader  = TestDataset(args.data_root, args.train_size, args.clip_size)

    output_path = args.resume

    print("Start eval...")    

    if args.task == 'Qualitative':
        Qualitative(test_loader, model, output_path)
    else:
        Qualitative(test_loader, model, output_path)
        
        