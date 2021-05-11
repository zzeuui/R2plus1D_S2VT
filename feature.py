import argparse
import os

from tqdm import tqdm

from utils.extractor.config import Config as c
from utils.extractor.feature_extractor import FeatureExtractor3D

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video_dpath', type=str, default='./data/youtube_videos', help="The directory path of videos.")
    parser.add_argument('-m', '--model', type=str, default='Default', help="The name of model from which you extract features.")
    parser.add_argument('-b', '--batch_size', type=int, default=1, help="The batch size.")
    parser.add_argument('-c', '--clip_num', type=int, default=20, help='the number of clip')
    parser.add_argument('-f', '--frame_num', type=int, default=32, help='the number of frame')
    parser.add_argument('-s', '--stride', type=int, default=32, help="Extract feature from every <s> frames.")
    parser.add_argument('-p', '--save_path', type=str, default='./features/dddddd', help="save path")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    print('clip num:', args.clip_num)
    print('frame num:', args.frame_num)
    print('stride:', args.stride)
    print('save path:', args.save_path)

    try:
        if not(os.path.isdir(args.save_path)):
            os.makedirs(os.path.join(args.save_path))
    except OSError as e:
        if e.errno != e.errno.EEXIST:
            print("failed to create directory")
            raise

    model = torch.hub.load("moabitcoin/ig65m-pytorch", 'r2plus1d_34_32_ig65m', num_classes=359, pretrained=True)
    model.cuda()
    model.eval()

    half_frame_num = int(args.frame_num/2)

    extractor = FeatureExtractor3D(
        clip_num=args.clip_num,
        frame_num=half_frame_num,
        stride=args.stride,
        mean=c.mean,
        std=c.std,
        resize_to=c.resize_to,
        crop_to=c.crop_to,
        model=model,
        batch_size=args.batch_size)

    videos = os.listdir(args.video_dpath)

    for video in tqdm(videos):
        video_fpath = os.path.join(args.video_dpath, video)

        feats = extractor(video_fpath, video, args.save_path)

