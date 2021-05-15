import cv2
import numpy as np
import os

for ty in ['train/', 'test/']:
    mp4_files = os.listdir('../dataset/minerl_navigate_mp4/' + ty)
    print("Train videos:", len(mp4_files))

    for mp4_file in mp4_files:
        if mp4_file[-4:] != '.mp4':
            continue
        vidcap = cv2.VideoCapture('../dataset/minerl_navigate_mp4/' + ty + mp4_file)
        images = []
        success,image = vidcap.read()
        while success:
            images.append(image)
            success,image = vidcap.read()
        images = np.array(images)
        np.save('../dataset/minerl_navigate/' + ty + mp4_file[:-4] + '.npy', images)
