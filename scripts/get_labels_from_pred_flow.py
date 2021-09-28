import sys
sys.path.append('..')
from helpers.flow_utils import *
import argparse
from os.path import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='mmnist')
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--vid', type=int, default=1)
    parser.add_argument('--jump', type=int, default=10000)

    args = parser.parse_args()

    # 10000 MMNIST videos
    video_nums = np.arange(args.vid, args.vid+args.jump).tolist()

    if args.dataset == 'mmnist':
        pred_flow_video_rootdir = '/lustre04/scratch/jithen/datasets/FlowNet_vis/MMNIST/inference/run.epoch-0-flow-vis'
        save_label_dir = '/lustre04/scratch/jithen/datasets/flow_labels/MMNIST'
    
    else:
        NotImplementedError('Have not implemented this script for the ' + args.dataset + ' dataset')
    
    k = args.topk # select top k grids with max avg. motion magnitude
    num_grids = k * k

    for video_num in video_nums:
        video_filename = 'video_' + str(video_num) + '.mp4'
        full_video_path = join(pred_flow_video_rootdir, video_filename)
        frames = get_video_frames(full_video_path)
        print("Video ", video_num)

        topk_motion_mags = [[0] * num_grids]
        motion_mag_bools = [[0] * num_grids]
        
        for i in range(len(frames)):
            frame = frames[i]
            topk_motion_mag, motion_mag_bool = get_avg_motion_mag_bool_for_frame(frame, num_grids, k)
            topk_motion_mags.append(topk_motion_mag)
            motion_mag_bools.append(motion_mag_bool)
        
        topk_motion_mags = np.array(topk_motion_mags)
        motion_mag_bools = np.array(motion_mag_bools)

        np_file_name = join(save_label_dir, 'video_' + str(video_num) + '.npy')
        np.save(np_file_name, motion_mag_bools)