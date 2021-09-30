import numpy as np
import os
import random
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import helpers.utils as utils

class MovingMNIST(Dataset):
    def __init__(self, root, is_train, n_frames_input, n_frames_output, num_objects,
                 transform=None, instances=1e4, device=None, frozen=False, offset=0, ch=1):
        '''
        param num_objects: a list of number of possible objects.
        '''
        super(MovingMNIST, self).__init__()
        self.frozen = frozen
        self.dataset = None
        self.device = device
        self.offset = offset
        self.channels = ch
        self.flow_label_dir = '/lustre04/scratch/jithen/datasets/flow_labels/MMNIST'

        if is_train and self.frozen is False:
            self.mnist = utils.load_mnist(root)
        elif self.frozen is False:
            if num_objects[0] != 2:
                self.mnist = utils.load_mnist(root)
            else:
                self.dataset = utils.load_fixed_set(root, False)

        self.root = root
        self.length = int(instances) if self.dataset is None else self.dataset.shape[1]

        self.is_train = is_train
        self.num_objects = num_objects
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform
        # For generating data
        self.image_size_ = 64
        self.digit_size_ = 28
        self.step_length_ = 0.1

    def get_random_trajectory(self, seq_length):
        ''' Generate a random sequence of a MNIST digit '''
        canvas_size = self.image_size_ - self.digit_size_
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)
        for i in range(seq_length):
            # Take a step along velocity.
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            # Bounce off edges.
            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y
            start_y[i] = y
            start_x[i] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def generate_moving_mnist(self, num_digits=2):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        data = np.zeros((self.n_frames_total, self.image_size_, self.image_size_), dtype=np.float32)
        for n in range(num_digits):
            # Trajectory
            start_y, start_x = self.get_random_trajectory(self.n_frames_total)
            ind = random.randint(0, self.mnist.shape[0] - 1)
            digit_image = self.mnist[ind]
            for i in range(self.n_frames_total):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.digit_size_
                right = left + self.digit_size_
                # Draw digit
                data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image)

        data = data[..., np.newaxis]
        return data

    def get_item(self, idx):
        
        frames = np.empty((200, 64, 64, self.channels), np.dtype('uint8'))
        count = 0
        video_filename = 'video_' + str(idx+1+self.offset) + '.mp4'
        if idx+1+self.offset < 8000:
            root = os.path.join(self.root, 'train')
        else:
            root = os.path.join(self.root, 'test')

        npy_filename = video_filename[:-4] + '.npy'

        video_filename = os.path.join(root, video_filename)
        vidcap = cv2.VideoCapture(video_filename)
        success, image = vidcap.read()
        while success is False: 
            print("retrying", count)
            vidcap = cv2.VideoCapture(video_filename)
            success, image = vidcap.read()
        
        while success:
            if self.channels == 1:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
                if gray.shape == (64, 64):  gray = gray.reshape(64, 64, 1)
                else:
                    print(gray.shape)
                    print(image.shape)
                frames[count] = gray
            else:
                frames[count] = image
            success, image = vidcap.read()
            count += 1
            retry = 0
            while success is False and count < 200: 
                retry += 1
                print("retrying", count, retry)
                if retry > 10:
                    break
                vidcap = cv2.VideoCapture(video_filename)
                success, image = vidcap.read()
                for _ in range(count+1):
                    success, image = vidcap.read()
        
        vidcap.release()
        cv2.destroyAllWindows()

        frames_in_video = count
        total_frames_to_sample = self.n_frames_total

        if count < 200:
            print(idx)
            print("frames_in_video < 200:", count)
            print(f"sampling from (0, {count - total_frames_to_sample + 1})")

        first_frame = np.random.randint(0, count - total_frames_to_sample + 1)
        
        if frames_in_video < 200 or first_frame+total_frames_to_sample > 200:
            print(idx)
            print("first_frame:", first_frame, first_frame+total_frames_to_sample)
        
        required_frames = frames[first_frame:first_frame+total_frames_to_sample]
        # ignore pos 0 which is a dummy flow label and get the flows corresponding to required_frames
        flow_labels = np.load(os.path.join(self.flow_label_dir, npy_filename))[first_frame + 1: first_frame + total_frames_to_sample]

        in_frames = required_frames[:self.n_frames_input]
        in_flow_labels = flow_labels[:self.n_frames_input-1]
        out_frames = required_frames[self.n_frames_input:]
        out_flow_labels = flow_labels[- (self.n_frames_input - 1):]
        in_frames = torch.from_numpy( (in_frames / 255.0) - 0.5 ).contiguous().float().to(self.device).permute(0, 3, 1, 2)
        out_frames = torch.from_numpy( (out_frames / 255.0) - 0.5 ).contiguous().float().to(self.device).permute(0, 3, 1, 2)
        
        mask = torch.ones((total_frames_to_sample, 1))
        mask = mask.type(torch.FloatTensor).to(self.device)
        
        out = {
            "idx": idx, 
            "observed_data": in_frames, 
            "data_to_predict": out_frames,
            'mask': mask,
            'in_flow_labels': in_flow_labels,
            'out_flow_labels': out_flow_labels}

        return out

    def __getitem__(self, idx):
        
        if self.frozen is True:
            
            if self.is_train is True:
                if idx > 8000: idx = idx % 8000
            else:
                if idx > 2000: idx = idx % 2000
            
            out = self.get_item(idx)
            return out

        length = self.n_frames_input + self.n_frames_output
        if self.is_train or self.num_objects[0] != 2:
            # Sample number of objects
            num_digits = random.choice(self.num_objects)
            # Generate data on the fly
            images = self.generate_moving_mnist(num_digits)
        else:
            images = self.dataset[:, idx, ...]

        r = 1
        w = int(64 / r)
        images = images.reshape((length, w, r, w, r)).transpose(0, 2, 4, 1, 3).reshape((length, r * r, w, w))
        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:length]
        else:
            output = []

        output = torch.from_numpy((output / 255.0) - 0.5).contiguous().float()
        input = torch.from_numpy((input / 255.0) - 0.5).contiguous().float()
        out = {
            "idx": idx, 
            "observed_data": input.to(self.device), 
            "data_to_predict": output.to(self.device), 
            "zeros": np.zeros(1)}
        
        return out

    def __len__(self):
        return self.length

def parse_datasets(opt, device):
    if opt.dataset == 'mmnist':
        total_frames = opt.total_frames # 2M as in clockwork paper
        total_instances = 1e4
        train_instances = int(opt.train_test_split * total_instances)       # 8000
        test_instances = int(total_instances - train_instances)                  # 2000
        print("Train frames:", opt.test_seq * train_instances)
        print("Test frames:", opt.test_seq * test_instances)
        print(f"Train instances {train_instances}; Test instances {test_instances}")
        
        trainFolder = MovingMNIST(is_train=True, root=opt.data_dir, n_frames_input=opt.train_in_seq, n_frames_output=opt.train_out_seq, num_objects=[opt.num_digits], instances=train_instances, device=device, frozen=opt.frozen, ch=opt.in_channels)
        testFolder = MovingMNIST(is_train=False, root=opt.data_dir, n_frames_input=opt.test_in_seq, n_frames_output=opt.test_out_seq, num_objects=[opt.num_digits], instances=test_instances, device=device, frozen=opt.frozen, offset=8000, ch=opt.in_channels)
    
        train_dataloader = DataLoader(trainFolder, batch_size=opt.batch_size, shuffle=False)
        test_dataloader = DataLoader(testFolder, batch_size=opt.batch_size, shuffle=False)

    else:
        raise NotImplementedError(f"There is no dataset named {opt.dataset}")

    data_objects = {"train_dataloader": utils.inf_generator(train_dataloader),
                    "test_dataloader": utils.inf_generator(test_dataloader),
                    "n_train_batches": len(train_dataloader),
                    "n_test_batches": len(test_dataloader)}
    
    return data_objects