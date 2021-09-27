import os, argparse
from os.path import *
import cv2

rootdir = '/lustre04/scratch/jithen/datasets/FlowNet_vis/MMNIST/inference/run.epoch-0-flow-vis'
parser = argparse.ArgumentParser()
parser.add_argument('--vid', type=int, default=1)
parser.add_argument('--jumps', type=int, default=2)


args = parser.parse_args()

video_file_names = []

start_vid_num = args.vid + 1

for i in range(args.jumps):
    vid_num = str(start_vid_num + i)
    video_folder = join(rootdir, 'video_' + vid_num)
    video_file = join(rootdir, 'video_' + vid_num + '.mp4')

    # Get flow png paths
    pngs = os.listdir(video_folder)
    pngs = sorted([png for png in pngs if png.endswith('png')])

    # Read flow pngs into 'frames'
    frames = [cv2.imread(join(video_folder, png)) for png in pngs]
    assert len(frames) == 199
    height, width, layers = frames[0].shape
    size = (width,height)

    # Convert 'frames' to video
    out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'DIVX'), 25.0, size)
    for i in range(len(frames)):
        out.write(frames[i])
    
    cv2.destroyAllWindows()
    out.release()
    print("Saved video at:", video_file)
    print('rm -rf ' + video_folder)
    # os.system('rm -rf ' + video_folder)
    # print(pngs)
    # print(video_folder)
    # print(len(os.listdir(video_folder)))
    