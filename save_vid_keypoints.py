import argparse
import logging
import time
import json
from tqdm import tqdm

import cv2
import numpy as np
import os

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from pytube import YouTube
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--video', type=str, default='video_test')

    parser.add_argument('--video_link', type=str, default='https://www.youtube.com/watch?v=Sdy5ghAD0Mo')

    parser.add_argument('--clip', type=str, default='0,0')

    parser.add_argument('--resize', type=str, default='432x368',
                        help='if provided, resize images before they are processed. default=, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()


    yt = YouTube(args.video_link)
    yt.streams.filter(file_extension='mp4').first().download('videos', filename=args.video)

    if args.clip != '0,0':
        os.rename('videos/' + args.video + '.mp4', 'videos/tmp.mp4')
        start = int(args.clip[:args.clip.find(',')])
        end = int(args.clip[args.clip.find(',')+1:])
        print(start, end)
        ffmpeg_extract_subclip('videos/tmp.mp4', start, end, targetname='videos/' + args.video + '.mp4')
        os.remove('videos/tmp.mp4')

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')
    cam = cv2.VideoCapture('videos/' + args.video + '.mp4')
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    logger.info('FPS: ' + str(cam.get(cv2.CAP_PROP_FPS)))

    with open('videos/' + args.video + '.fps', 'w+') as f:
        f.write(str(cam.get(cv2.CAP_PROP_FPS)))

    n_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    all_cords = []
    for _ in tqdm(range(n_frames)):
        ret_val, image = cam.read()
        if not ret_val:
            break

        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        all_cords.append(TfPoseEstimator.get_cords(image, humans, imgcopy=False))

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    print(len(all_cords))
    print(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    with open('keypoints/' + args.video + '.keypoints.json', 'w+') as f:
        json.dump(all_cords, f)
