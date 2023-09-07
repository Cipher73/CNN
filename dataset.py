import random
import glob
import pandas as pd
import torch
import os
import torchvision.transforms as transforms
import torch.distributions as dist
import tqdm
from PIL import Image
from torch.utils.data import Dataset
import multiprocessing as mp
import argparse
import matplotlib.pyplot as plt
import cv2
import numpy as np

import random

# Define a default transformation pipeline using torchvision.transforms.Compose
default_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor(),
])

# Custom data augmentation transformations

# Adds shadow artifacts to images
class ShadowArtifacts(object):
    def __init__(self, shadow_intensity=0.9, shadow_size=0.7):
        self.shadow_intensity = shadow_intensity
        self.shadow_size = shadow_size

    def __call__(self, tensor):
        h, w = tensor.shape[-2:]
        shadow_mask = torch.ones_like(tensor)
        start = int((1 - self.shadow_size) * h)
        end = h
        shadow_mask[:, start:end, :] = self.shadow_intensity
        return tensor * shadow_mask

    def __repr__(self):
        return self.__class__.__name__ + '(shadow_intensity={0}, shadow_size={1})'.format(self.shadow_intensity, self.shadow_size)

# Simulates motion artifacts by applying random blurring
class MotionArtifacts(object):
    def __init__(self, max_blur=5):
        self.max_blur = max_blur

    def __call__(self, tensor):
        blur_amount = random.randint(1, self.max_blur)
        blurred_tensor = torch.tensor(tensor).numpy()
        for _ in range(blur_amount):
            direction = random.choice(['horizontal', 'vertical'])
            kernel_size = random.randint(3, 10)
            if direction == 'horizontal':
                kernel = np.ones((1, kernel_size)) / kernel_size
            else:
                kernel = np.ones((kernel_size, 1)) / kernel_size
            blurred_tensor = cv2.filter2D(blurred_tensor, -1, kernel)
        return torch.tensor(blurred_tensor)

    def __repr__(self):
        return self.__class__.__name__ + '(max_blur={0})'.format(self.max_blur)

# Adds additive speckle noise to images
class AdditiveSpeckleNoise(object):
    def __init__(self, variance=0.1):
        self.variance = variance

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * (self.variance ** 0.5) + 1
        return torch.clamp(tensor * noise, 0., 1.)

    def __repr__(self):
        return self.__class__.__name__ + '(variance={0})'.format(self.variance)

# Reduces image contrast
class LowContrastNoise(object):
    def __init__(self, scale=0.8):
        self.scale = scale

    def __call__(self, tensor):
        noisy_tensor = tensor * self.scale
        return torch.clamp(noisy_tensor, 0., 1.)

    def __repr__(self):
        return self.__class__.__name__ + '(scale={0})'.format(self.scale)

# Adds additive Gaussian white noise to images
class AdditiveGaussianWhiteNoise(object):
    def __init__(self, mean=0., std=50.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        noise = (torch.randn(tensor.size()) * self.std + self.mean) / 255.
        return torch.clamp(tensor + noise, 0., 1., )

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# Simulates signal attenuation
class SignalAttenuation(object):
    def __init__(self, attenuation_factor=0.5):
        self.attenuation_factor = attenuation_factor

    def __call__(self, tensor):
        attenuated_tensor = tensor * self.attenuation_factor
        return torch.clamp(attenuated_tensor, 0., 1.)

    def __repr__(self):
        return self.__class__.__name__ + '(attenuation_factor={0})'.format(self.attenuation_factor)

# Identity transformation
class NoopTransform(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor

    def __repr__(self):
        return self.__class__.__name__

# Function to generate random noise levels for SIDD dataset
def random_noise_levels_sidd():
    log_min_shot_noise = torch.log10(torch.Tensor([0.0001]))
    log_max_shot_noise = torch.log10(torch.Tensor([0.012]))
    distribution = dist.uniform.Uniform(log_min_shot_noise, log_max_shot_noise)

    log_shot_noise = distribution.sample()
    shot_noise = torch.pow(10, log_shot_noise)
    distribution = dist.normal.Normal(torch.Tensor([0.0]), torch.Tensor([0.26]))
    read_noise = distribution.sample()
    line = lambda x: 2.18 * x + 1.20
    log_read_noise = line(log_shot_noise) + read_noise
    read_noise = torch.pow(10, log_read_noise)
    return shot_noise, read_noise

# List of different noise types and their corresponding noise generators
noise_types = [
    {
        'name': 'shadow',
        'noise_generator': ShadowArtifacts()
    },
    {
        'name': 'motion',
        'noise_generator': MotionArtifacts()
    },
    {
        'name': 'speckle',
        'noise_generator': AdditiveSpeckleNoise()
    },
    {
        'name': 'gaussian_white',
        'noise_generator': AdditiveGaussianWhiteNoise()
    },
    {
        'name': 'low_contrast',
        'noise_generator': LowContrastNoise()
    },
    {
        'name': 'signal_attenuation',
        'noise_generator': SignalAttenuation()
    },
    {
        'name': 'noop',
        'noise_generator': NoopTransform()
    }
]

# Noise percentages for training, validation, and test datasets
noise_percentages_train = [0.25, 0.15, 0.25, 0.1, 0.1, 0.05, 0.1]
noise_percentages_val = [0.2, 0.15, 0.3, 0.15, 0.1, 0.05, 0.05]
noise_percentages_test = [0.15, 0.15, 0.4, 0.15, 0.05, 0.05, 0.05]

# Function to shuffle a list of images
def shuffle_data(images):
    random.seed(42)
    random.shuffle(images)
    return images

# Function to add random shot and read noise to images
def add_noise(noise_func=random_noise_levels_sidd):
    def _func(image):
        shot_noise, read_noise = random_noise_levels_sidd()
        variance = image * shot_noise + read_noise
        mean = torch.Tensor([0.0])
        distribution = dist.normal.Normal(mean, torch.sqrt(variance))
        noise = distribution.sample()
        return torch.clamp(image + noise, 0, 1)
    return _func

# Create a noise model with random shot and read noise
fake_noise_model = add_noise()

class CustomDataset(Dataset):
    def __init__(self, root_dir, data_type='train', transform=default_transform,load_fake=False,noise_generator=fake_noise_model,noise_name='gaussian'):
        self.root_dir = root_dir
        self.data_type = data_type
        self.transform = transform
        self.image_dir = f'{root_dir}/{data_type}/images/'
        self.mask_dir = f'{root_dir}/{data_type}/masks/'
        self.image_paths = sorted(os.listdir(self.image_dir))
        self.noise_generator = noise_generator
        self.load_fake = load_fake
        self.noise_name = noise_name
        self.transform = transform
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        #shuffle_data(self.image_paths)
        image_path = os.path.join(self.image_dir, self.image_paths[idx])

        image = Image.open(image_path)
        mask_path = os.path.join(self.mask_dir, self.image_paths[idx])
        mask = Image.open(mask_path)


        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

    def process_fake_noise_image(self, work_id, worker_num):
        process = tqdm.tqdm(range(work_id, len(self.image_paths), worker_num))
        for idx in process:
            process.set_description(f'Worker Id: {work_id}')
            image_path = os.path.join(self.image_dir, self.image_paths[idx])
            mask_path = os.path.join(self.mask_dir, self.image_paths[idx])

            
            # Load the clean image and mask
            clean = Image.open(image_path)
            
            mask = Image.open(mask_path)
           
            
            # Apply transformations to the clean image and mask
            if self.transform:
                state = torch.get_rng_state()
                clean = self.transform(clean)
                torch.set_rng_state(state)
            
            # Generate fake noisy image from the clean image
            fake_noisy = self.noise_generator(clean)
            
            # Save the generated data
            noise_name = self.noise_name
            filename = noise_name + "_" + self.image_paths[idx]
            fake_noisy = transforms.ToPILImage()(fake_noisy)
            fake_noisy.save(os.path.join(self.root_dir, self.data_type, 'noisy_images', filename))  # Save the image
            mask.save(os.path.join(self.root_dir, self.data_type, 'noisy_masks', filename))  # Save the mask


        process.set_description(f'Worker {work_id} Finished Job')
        process.close()

    def generate_noise_image(self, workers):
        print(f'Start Processing with {workers} workers')
        noisy_crops_dir = os.path.join(self.root_dir, self.data_type, 'noisy_images')
         
        if not os.path.exists(noisy_crops_dir):
            os.makedirs(noisy_crops_dir)
        process_pool = []
        for i in range(workers):
            proc = mp.Process(target=self.process_fake_noise_image, args=(i, workers,))
            process_pool.append(proc)
            proc.start()
        for each in process_pool:
            each.join()
        print('Finished')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='Dataset', help='root dir')
    parser.add_argument('--workers', type=int, default=4, help='How many workers work together')
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL Image to PyTorch tensor
    ])
     


    train_datasets_with_noises = []  
    val_datasets_with_noises = []   
    test_datasets_with_noises = []

    for noise_type in noise_types:
        noise_generator = noise_type['noise_generator']
        noise_name = noise_type['name']

        train_dataset = CustomDataset(root_dir=args.root, data_type='train', transform=transform, load_fake=False, noise_generator=noise_generator, noise_name=noise_name)
        val_dataset = CustomDataset(root_dir=args.root, data_type='val', transform=transform, load_fake=False, noise_generator=noise_generator, noise_name=noise_name)
        test_dataset = CustomDataset(root_dir=args.root, data_type='test', transform=transform, load_fake=False, noise_generator=noise_generator, noise_name=noise_name)

        train_datasets_with_noises.append(train_dataset)
        val_datasets_with_noises.append(val_dataset)
        test_datasets_with_noises.append(test_dataset)

        
    print('Start To process for Train Dataset')
    for train_dataset in train_datasets_with_noises:
        train_dataset.generate_noise_image(args.workers)

    print('Start To process for Val Dataset')
    for val_dataset in val_datasets_with_noises:
        val_dataset.generate_noise_image(args.workers)

    print('Start To process for Test Dataset')
    for test_dataset in test_datasets_with_noises:
        test_dataset.generate_noise_image(args.workers)

