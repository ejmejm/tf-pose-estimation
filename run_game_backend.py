import json
import argparse
import logging
import time
import threading
import subprocess

import cv2
import numpy as np
import sys

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
global_score = 0

score_file_idx = 0
fps_time = 0
angle_cutoff = 25 # In degrees
angle_max_score = 1

def dist(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def get_angle(p1, p2, p3):
    """Returns angle in degrees"""
    p21 = dist(p2, p1)
    p23 = dist(p2, p3)
    p13 = dist(p1, p3)

    return np.arccos((p21**2 + p23**2 - p13**2)/(2 * p21 * p23)) * (180./np.pi)

def angle_err_to_score(angle_err):
    if angle_err > angle_cutoff:
        return 0
    return ((angle_cutoff**2 - angle_err**2) / angle_cutoff**2) * angle_max_score

def get_pose_match_score(pose1, pose2, part1, part2, part3):
    try:
        a = get_angle(pose1[part1], pose1[part2], pose1[part3])
        b = get_angle(pose2[part1], pose2[part2], pose2[part3])

        return angle_err_to_score(abs(a - b))
    except KeyError:
        return 0

def get_pose_sim(pose1, pose2):
    score = 0

    score += get_pose_match_score(pose1, pose2, 'lshoulder', 'lelbow', 'lwrist')
    score += get_pose_match_score(pose1, pose2, 'rshoulder', 'relbow', 'rwrist')

    score += get_pose_match_score(pose1, pose2, 'lhip', 'lshoulder', 'lelbow')
    score += get_pose_match_score(pose1, pose2, 'rhip', 'rshoulder', 'relbow')

    score += get_pose_match_score(pose1, pose2, 'lankle', 'lhip', 'lshoulder') * 0.1
    score += get_pose_match_score(pose1, pose2, 'rankle', 'rhip', 'rshoulder') * 0.1

    score += get_pose_match_score(pose1, pose2, 'lankle', 'lknee', 'lhip') * 0.1
    score += get_pose_match_score(pose1, pose2, 'rankle', 'rknee', 'rhip') * 0.1

    return score

def get_frame(frame_data, frame_idx):
    """Indexed from 0"""
    return frame_data[frame_idx][0]

def calc_score(all_data, recent_frames, start_time, fps):
    global score_file_idx
    total_players = 2
    # for frame in recent_frames:
    #     total_players = max(total_players, len(frame[1]))

    scores = [0] * max(total_players, 1)

    for frame in recent_frames:
        frame_ts = frame[0]
        frame_humans = frame[1]
        secs_passed = frame_ts - start_time
        frame_idx = int(secs_passed * fps)
        # frame_idx = int((secs_passed / total_secs) * total_frames)
        human_idx = 0
        for human in frame_humans:
            if len(all_data[frame_idx]) > 0:
                pose_sim = get_pose_sim(human, all_data[frame_idx][0])
                if pose_sim is not np.nan:
                    scores[human_idx] += pose_sim
                human_idx += 1
    # print(scores[0])

    global global_score
    # with open('score_update_%d.txt' % score_file_idx, 'w+') as f:
    #     # scores_string = ''
    #     # for score in scores:
    #     #     scores_string += str(score) + ', '
    #     # f.write(scores_string[:-2])
    #     f.write(str(scores[0]))
    if scores[0] > 0:
        global_score += scores[0]

    score_file_idx += 1

def play_video(start_time, vid_path, fps):
    cap = cv2.VideoCapture(vid_path)

    frame_idx = 0

    while True:
        #if frame_idx * (1./fps) < time.time() - start_time:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.imshow('frame',frame)
            cv2.waitKey(int(1./fps*1000))
            frame_idx += 1
        #time.sleep(1)

    cap.release()
    cv2.destroyAllWindows()

def play_video2(start_time, vid_path, fps):
    subprocess.call(['ffplay', '-loglevel', 'panic', vid_path])

if len(sys.argv) > 1:
    choreo_keypoints_path = 'keypoints/' + sys.argv[1] + '.keypoints.json'
    video_path = 'videos/' + sys.argv[1] + '.mp4'
    with open('videos/' + sys.argv[1] + '.fps') as f:
        fps = float(f.read().strip())
else:
    choreo_keypoints_path = 'keypoints/test_video.keypoints.json'
    video_path = 'videos/test_video.mp4'
    fps = 30

# tmp = cv2.VideoCapture('video_test.mp4')
# tmp.set(cv2.CAP_PROP_POS_MP4_RATIO,1)
# s_len = tmp.get(cv2.CAP_PROP_POS_MSEC) / 1000.

#play_video2(start_time, 'video_test.mp4', fps)

args_model = 'mobilenet_thin'

logger.debug('initialization %s : %s' % (args_model, get_graph_path(args_model)))
w, h = model_wh('432x368')
if w > 0 and h > 0:
    e = TfPoseEstimator(get_graph_path(args_model), target_size=(w, h))
else:
    e = TfPoseEstimator(get_graph_path(args_model), target_size=(432, 368))
cam = cv2.VideoCapture(0)
ret_val, image = cam.read()

choreo_json = json.load(open(choreo_keypoints_path, 'r'))
start_time = time.time()
threading.Thread(target=play_video2, args=(start_time, video_path, fps)).start()
time.sleep(1)

print('NUM OF FRAME:', len(choreo_json))
print('VIDEO PATH:', video_path)
print('KEYPOINTS PATH:', choreo_keypoints_path)

sec_start_time = time.time()
curr_cords = []
last_cords = [time.time(), [{}]]
part_linger_max = 3
part_linger = {}

while time.time() - start_time + 1 < len(choreo_json) / fps:
    ret_val, image = cam.read()

    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.)

    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    curr_cords.append([time.time(), TfPoseEstimator.get_cords(image, humans, imgcopy=False)])
    print(last_cords, 'a')
    if len(last_cords[1]) == 0:
        last_cords[1].append({})
    for key in last_cords[1][0].keys():
        if len(curr_cords[-1][1]) == 0:
            curr_cords[-1][1].append({})
        if key not in curr_cords[-1][1][0]:
            if key not in part_linger:
                part_linger[key] = 0
            part_linger[key] += 1
            if part_linger[key] > 3:
                part_linger[key] = 0
            else:
                curr_cords[-1][1][0][key] = last_cords[1][0][key]
    last_cords = curr_cords[-1]

    # try:
    #     if len(curr_cords[-1][1]) > 0:
    #         print(get_angle(curr_cords[-1][1][0]['lhip'], curr_cords[-1][1][0]['lshoulder'], curr_cords[-1][1][0]['lelbow']))
    # except:
    #    pass

    # cv2.putText(image,
    #             "FPS: %f" % (1.0 / (time.time() - fps_time)),
    #             (15, 15),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (0, 255, 0), 2)

                
    image = cv2.flip(image, 1)
    # try:
    #     cv2.putText(image,
    #                 str(get_angle(curr_cords[-1][1][0]['lhip'], curr_cords[-1][1][0]['lshoulder'], curr_cords[-1][1][0]['lelbow'])),
    #                 (15, 15),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #                 (0, 255, 0), 2)
    # except:
    #     pass

    cv2.putText(image,
                str(int(global_score)),
                (20, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (20, 20, 240), 6)

    cv2.imshow('tf-pose-estimation result', image)
    
    if time.time() - sec_start_time >= 1:
        nt = threading.Thread(target=calc_score, args=(choreo_json, curr_cords, start_time, fps))
        nt.start()
        sec_start_time = time.time()
        curr_cords = []

    if cv2.waitKey(1) == 27: # ESC
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
