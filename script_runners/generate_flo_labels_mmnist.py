import numpy as np
import os

# MMNIST has 10000 videos
videos = 10000
jump = 500

# Submit (videos / jump) jobs. Each job creates labels for `jump` jobs
start_idx = np.arange(0, videos, jump)
for idx in start_idx:
    start_video_num = idx + 1
    print()
    command = 'cd ../scripts && sbatch --time 0:30:00 --output ../out/Flo_labels/MMNIST/slurm-%j.out generate_flo_labels.sh ' + str(start_video_num) + ' ' + str(jump) + ' mmnist'
    print(command)
    os.system(command)
