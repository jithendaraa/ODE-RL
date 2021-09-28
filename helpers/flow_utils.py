import os
from os.path import *
import cv2
import numpy as np
from skimage.color import rgb2hsv


def get_video_frames(full_video_path, channels=3, max_frames=199):
    frames = np.empty((max_frames, 64, 64, channels), np.dtype('uint8'))
    count = 0

    vidcap = cv2.VideoCapture(full_video_path)
    success, image = vidcap.read()
    while success is False: 
        print("retrying", count)
        vidcap = cv2.VideoCapture(full_video_path)
        success, image = vidcap.read()
    
    while success:
        if channels == 1:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
            if gray.shape == (64, 64):  gray = gray.reshape(64, 64, 1)
            else:
                print(gray.shape)
                print(image.shape)
            frames[count] = gray
        else:
            frames[count] = image
        count += 1
        success, image = vidcap.read()
        if count == max_frames:
            assert success is False
            break
        
        retry = 0
        while success is False and count < max_frames: 
            retry += 1
            print("retrying", count, retry)
            if retry > 10:
                break
            vidcap = cv2.VideoCapture(full_video_path)
            success, image = vidcap.read()
            for _ in range(count+1):
                success, image = vidcap.read()
        

    vidcap.release()
    cv2.destroyAllWindows()
    return frames

def div_into_grids(frame, resolution=(3, 3)):
    sizeX, sizeY, c = frame.shape
    nRows, mCols = resolution
    patches = []

    for i in range(0,nRows):
        for j in range(0, mCols):
            start_X = int(i*sizeY/nRows)
            end_X = int(i*sizeY/nRows + sizeY/nRows)
            start_Y = int(j*sizeX/mCols)
            end_Y = int(j*sizeX/mCols + sizeX/mCols)

            if end_X == sizeX: end_X += 1
            if end_Y == sizeY: end_Y += 1

            roi = frame[start_X:end_X, start_Y:end_Y, :]
            patches.append(roi)
    
    return patches

def get_avg_motion_mag_bool_for_frame(frame, num_grids, k):
    patches = div_into_grids(frame)
    sats = []
    motion_mag_bool = [0] * num_grids

    for patch in patches:
        hsv_patch = rgb2hsv(patch)
        hue = hsv_patch[:, :, 0].mean()
        sat = hsv_patch[:, :, 1].mean()
        val = hsv_patch[:, :, 2].mean()
        sats.append(sat)

    sats = np.array(sats)
    topk_motion_mag = 1 + sats.argsort()[-k:][::-1]
    for v in sats.argsort()[-k:][::-1]:
        motion_mag_bool[v] = 1
    
    return topk_motion_mag, motion_mag_bool



        
