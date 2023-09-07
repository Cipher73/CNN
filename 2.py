
from torch.utils.data import Dataset
import os  
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
from catalyst.dl import SupervisedRunner
from segmentation_models_pytorch.utils.losses import DiceLoss,CrossEntropyLoss
from catalyst.dl import SupervisedRunner
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from catalyst.dl import CriterionCallback,PeriodicLoaderCallback

class CustomDataset(Dataset):
    def __init__(self, root_dir, data_type='train', transform=None):
        self.root_dir = root_dir
        self.data_type = data_type
        self.transform = transform  
        self.image_dir = f'{root_dir}/{data_type}/images/'
        self.mask_dir = f'{root_dir}/{data_type}/masks/'
        self.image_paths = sorted(os.listdir(self.image_dir))


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        #shuffle_data(self.image_paths)
        image_path = os.path.join(self.image_dir, self.image_paths[idx])

        image = Image.open(image_path)
        #convert to rgb 3,128,128
        image = image.convert('RGB')
        #print(image.size) #(128,128)
      
        mask_path = os.path.join(self.mask_dir, self.image_paths[idx])
        mask = Image.open(mask_path)


        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        result = {'image': image, 'mask': mask}
        return result
    

transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL Image to PyTorch tensor
])
    

train_dataset = CustomDataset(root_dir='Dataset', data_type='train', transform=transform)
val_dataset = CustomDataset(root_dir='Dataset', data_type='val', transform=transform)
test_dataset = CustomDataset(root_dir='Dataset', data_type='test', transform=transform)


#convert targets to tragers.


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True,num_workers=4)
test_loader = DataLoader(test_dataset,batch_size=16, shuffle=True,num_workers=4)


import torch
import segmentation_models_pytorch as smp
from torch import optim


# Define hyperparameters
learning_rate = 0.001
num_epochs = 10
num_classes = 10

# Initialize the model, optimizer, and loss function
model = smp.Unet('resnet50', classes=num_classes, activation='softmax')
# Define Loss and Metrics to Monitor (Make sure mode = "multiclass") ======================================
criterion = smp.losses.DiceLoss(mode="multiclass")
#loss_fn.__name__ = 'Dice_loss'

# Define Optimizerand learning rate ============================
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        inputs = batch['image']
        targets = batch['mask']

        optimizer.zero_grad()
        outputs = model(inputs)
        targets = targets.argmax(dim=1)  # Assuming targets are one-hot encoded
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Print or log training loss for this epoch
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {train_loss / len(train_loader)}")

# Optionally, you can save the trained model
torch.save(model.state_dict(), 'unet_model.pth')


# Optional: Load other components like optimizer, scheduler, etc., if needed
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# Ensure the model is in evaluation mode (if needed)
model.eval()

# Create lists to store images, ground truth masks, and predicted masks
images = []
gt_masks = []
pred_masks = []

# Loop through the test dataset and make predictions
with torch.no_grad():
    for batch in test_loader:
        inputs = batch['image']
        masks = batch['mask']

        # Forward pass
        outputs = model(inputs)

        # Convert tensors to numpy arrays
        inputs = inputs.cpu().numpy()
        masks = masks.cpu().numpy()
        outputs = outputs.cpu().numpy()

        # Append to lists
        images.append(inputs)
        gt_masks.append(masks)
        pred_masks.append(outputs)

# Convert lists to numpy arrays
images = np.concatenate(images, axis=0)
gt_masks = np.concatenate(gt_masks, axis=0)
pred_masks = np.concatenate(pred_masks, axis=0)

# Visualize some results
num_samples_to_visualize = 3

for i in range(num_samples_to_visualize):
    plt.figure(figsize=(12, 4))

    # Plot the input image
    plt.subplot(1, 3, 1)
    plt.imshow(images[i].transpose(1, 2, 0))
    plt.title("Input Image")

    # Plot the ground truth mask
    plt.subplot(1, 3, 2)
    plt.imshow(gt_masks[i][0])
    plt.title("Ground Truth Mask")

    # Plot the predicted mask
    plt.subplot(1, 3, 3)
    plt.imshow(pred_masks[i][0])
    plt.title("Predicted Mask")

    # Save the current figure
    plt.savefig(f"sample_{i+1}.png")
    plt.close()  # Close the figure to release resources

print("Plots saved successfully.")
