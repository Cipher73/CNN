import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import glob
from sklearn.metrics import jaccard_score
import torch
import torch.nn as nn
import torchvision.models as models

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Convolution layers to adjust the number of channels
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.conv3x3_4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.conv3x3_5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)

    def forward(self, x):
        x1 = self.conv1x1_1(self.global_avg_pool(x))
        print(x1.shape)
        x1 = F.interpolate(x1, size=x.size()[2:], mode='bilinear', align_corners=True)  # Upscale x1

        x2 = self.conv1x1_2(x)
        x3 = self.conv3x3_3(x)
        x4 = self.conv3x3_4(x)
        x5 = self.conv3x3_5(x)
  
        
        # Ensure all tensors have the same number of channels
        x2 = x2 if x2.size()[2:] == x1.size()[2:] else F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=True)
        
        # Concatenate the adjusted tensors
        out = torch.cat((x1, x2, x3, x4, x5), dim=1)
        # convert to 256 channels
        out = nn.Conv2d(5*256, 256, kernel_size=1)(out)
        return out


class DeepLabV3Plus(nn.Module):
    def __init__(self, backbone, num_classes):
        super(DeepLabV3Plus, self).__init__()

        # Backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            in_channels = 2048
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
            in_channels = 512
        else:
            raise ValueError("Backbone must be 'resnet50' or 'resnet34'.")

        # Remove the fully connected layer and average pooling layer from the backbone
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # ASPP (Atrous Spatial Pyramid Pooling)
        self.aspp = ASPP(in_channels, 256)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Backbone
        x = self.backbone(x)

        # ASPP
        x = self.aspp(x)

        # Decoder
        x = self.decoder(x)

        return x

# Example usage:
# model = DeepLabV3Plus(backbone='resnet50', num_classes=num_classes)

# Example usage:
# deep_lab_v3plus_resnet50 = DeepLabV3Plus(backbone='resnet50', num_classes=21)
# deep_lab_v3plus_resnet34 = DeepLabV3Plus(backbone='resnet34', num_classes=21)


class DeepLabV3(nn.Module):
    #read the paper online!!
    def __init__(self, backbone, num_classes):
        super(DeepLabV3, self).__init__()

        # Backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
        else:
            raise ValueError("Backbone must be 'resnet50' or 'resnet34'.")

        # Remove the fully connected layer and average pooling layer from the backbone
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # ASPP (Atrous Spatial Pyramid Pooling)
        self.aspp = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=12),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=18),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Backbone
        x = self.backbone(x)

        # ASPP
        x = self.aspp(x)

        # Decoder
        x = self.decoder(x)

        return x

# Example usage:
# deep_lab_resnet50 = DeepLabV3(backbone='resnet50', num_classes=21)
# deep_lab_resnet34 = DeepLabV3(backbone='resnet34', num_classes=21)

class UNet(nn.Module):
    def __init__(self, backbone, in_channels, out_channels):
        super(UNet, self).__init__()

        # Backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=False)
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=False)
        else:
            raise ValueError("Backbone must be 'resnet50' or 'resnet34'.")

        # Remove the fully connected layer and average pooling layer from the backbone
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1),  # Adjust the number of channels based on the backbone
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # Backbone
        x = self.backbone(x)
        # Decoder
        x = self.decoder(x)
        return x


class DataProcessor:
    @staticmethod
    def get_data(directory_path, flag):
        images = []
        for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
            img = cv2.imread(img_path, flag)
            images.append(img)
        images = np.array(images)
        return images

    @staticmethod
    def shuffle_data(images, masks):
        images, masks = shuffle(images, masks, random_state=0)
        return images, masks

    def preprocess_data(self, train_images, train_masks, val_images, val_masks, test_images, test_masks):
        train_images = np.array(train_images)
        train_masks = np.array(train_masks)
        val_images = np.array(val_images)
        val_masks = np.array(val_masks)
        test_images = np.array(test_images)
        test_masks = np.array(test_masks)

        # Label encoding for training masks
        labelencoder = LabelEncoder()
        n, h, w = train_masks.shape
        train_masks_reshaped = train_masks.reshape(-1, 1)
        train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
        train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

        X_train = train_images
        y_train = train_masks_encoded_original_shape
        X_test = test_images
        y_test = test_masks
        X_val = val_images
        y_val = val_masks
        X_train = train_images.transpose(0, 3, 1, 2)  # Transpose image dimensions
        y_train = train_masks_encoded_original_shape
        X_test = test_images.transpose(0, 3, 1, 2)    # Transpose image dimensions
        y_test = test_masks
        X_val = val_images.transpose(0, 3, 1, 2)      # Transpose image dimensions
        y_val = val_masks
        n_classes = len(np.unique(y_train))
        y_train_cat = torch.tensor(y_train, dtype=torch.long)
        y_test_cat = torch.tensor(y_test, dtype=torch.long)
        y_val_cat = torch.tensor(y_val, dtype=torch.long)
        print(X_train[0].shape, y_train_cat[0].shape)
        #X_train = X_train.astype(np.float64)
        return X_train, y_train_cat, X_test, y_test_cat, X_val, y_val_cat, n_classes
    
def calculate_iou(predictions, targets):
    # Flatten predictions and targets
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    # Calculate IoU
    intersection = (predictions & targets).sum().float()
    union = (predictions | targets).sum().float()
    iou = (intersection / union).item()
    return iou

# Training function
def train(model, train_loader, val_loader, num_epochs, criterion, optimizer,save_name):
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            # Forward pass
            outputs = model(inputs)

            # Resize targets to match the output size
            targets = torch.nn.functional.interpolate(targets.unsqueeze(1).float(), size=outputs.shape[-2:], mode='nearest').squeeze(1).long()

            # Calculate the loss
            loss = criterion(outputs, targets)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Validation loop
        model.eval()
        with torch.no_grad():
            total_val_loss = 0.0
            total_iou = 0.0
            for inputs, targets in val_loader:
                # Forward pass
                outputs = model(inputs)

                # Resize targets to match the output size
                targets = torch.nn.functional.interpolate(targets.unsqueeze(1).float(), size=outputs.shape[-2:], mode='nearest').squeeze(1).long()

                # Calculate the loss
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

                # Calculate IoU
                iou = calculate_iou((outputs.argmax(dim=1) == 1).cpu(), (targets == 1).cpu())
                total_iou += iou

            # Calculate and print validation loss and IoU
            avg_val_loss = total_val_loss / len(val_loader)
            avg_iou = total_iou / len(val_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {total_train_loss:.4f} - Validation Loss: {avg_val_loss:.4f} - IoU: {avg_iou:.4f}")
    #save model
    torch.save(model.state_dict(), f"BuiltModels/{save_name}.pth")

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
def test(model, test_loader, save_path,save_name):
    #read model from file
    model.load_state_dict(torch.load(f"BuiltModels/{save_name}.pth"))
    
    model.eval()
    all_preds = []
    all_targets = []
    all_test_images = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)


    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs,targets = inputs.to(device), targets.to(device)
            all_test_images.append(inputs.cpu().numpy())
            print(inputs.shape)
            #reshape targets to match the output size
            targets = torch.nn.functional.interpolate(targets.unsqueeze(1).float(), size=(64,64), mode='nearest').squeeze(1).long()
            preds = model(inputs.float())
            print(preds.shape)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    X_test = np.concatenate(all_test_images)
    all_targets = np.concatenate(all_targets)
    y_pred_argmax = np.argmax(all_preds, axis=1)
    y_test_argmax = all_targets
    cm = confusion_matrix(y_test_argmax.flatten(), y_pred_argmax.flatten())

    # Normalize the confusion matrix
    cm_normalized = cm / cm.sum(axis=1)
    #check if save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Plot and save the confusion matrix as an image
    plt.imshow(cm_normalized, cmap='hot')
    plt.colorbar()
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Normalized Confusion Matrix')
    plt.savefig(f"{save_path}/confusion_matrix.png")
    plt.close()

    # Loop through some test images for visualization and logging
    for i in range(min(3, len(X_test))):
        # Clean up: Delete the temporary images if they are no longer needed
        # Save image ground truth and prediction on the same image
        plt.subplot(2,3,1)
        print(X_test[i].shape)
        #convert to grayscale
        plt.imshow(X_test[i][0],cmap='gray')
 
        plt.title('Image')
        plt.subplot(2,3,2)
        plt.imshow(y_test_argmax[i])
        plt.title('Ground Truth')
        plt.subplot(2,3,3)
        plt.imshow(y_pred_argmax[i])
        plt.title('Prediction')
        plt.savefig(f"{save_path}/test_{i}.png")
        plt.close()

    # Calculate accuracy
    accuracy = accuracy_score(y_test_argmax.flatten(), y_pred_argmax.flatten())
    model.train()

    # Print accuracy
    print(f"Accuracy: {accuracy:.4f}")



def main():
    data_processor = DataProcessor()

    # Load and preprocess data
    train_images = np.array(data_processor.get_data("Dataset/train/noisy_images/", 1))
    train_masks = np.array(data_processor.get_data("Dataset/train/noisy_masks/", 0))
    val_images = np.array(data_processor.get_data("Dataset/val/noisy_images/", 1))
    val_masks = np.array(data_processor.get_data("Dataset/val/noisy_masks/", 0))
    test_images = np.array(data_processor.get_data("Dataset/test/noisy_images/", 1))
    test_masks = np.array(data_processor.get_data("Dataset/test/noisy_masks/", 0))

    # Shuffle the data
    train_images, train_masks = data_processor.shuffle_data(train_images, train_masks)
    val_images, val_masks = data_processor.shuffle_data(val_images, val_masks)
    test_images, test_masks = data_processor.shuffle_data(test_images, test_masks)

    # Preprocess the data
    X_train, y_train_cat, X_test, y_test_cat, X_val, y_val_cat, n_classes = data_processor.preprocess_data(
        train_images, train_masks, val_images, val_masks, test_images, test_masks
    )


    # Define your UNet with Backbone model
    in_channels_backbone = 3  # Number of input channels (number of color channels in the image. 3 = RGB)
    out_channels_backbone = 11 # Number of output channels (number of classes)
    unet_model_resnet50 = UNet('resnet50', in_channels_backbone, out_channels_backbone)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(unet_model_resnet50.parameters(), lr=0.001)

    # Create DataLoader objects for training, validation, and testing
    batch_size = 32  # Adjust as needed
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), y_train_cat)
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), y_val_cat)
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), y_test_cat)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    num_epochs = 15

    # Training and testing for UNet with Backbone
    train(unet_model_resnet50, train_loader, val_loader, num_epochs, criterion, optimizer,"unet_resnet50")
    test(unet_model_resnet50, test_loader, save_path="test_results_unet_backbone",save_name="unet_resnet50")


    # DeepLabV3+ with ResNet50 backbone
    deeplabv3plus_resnet50 = DeepLabV3Plus(backbone='resnet50', num_classes=out_channels_backbone)
    train(deeplabv3plus_resnet50, train_loader, val_loader, num_epochs, criterion, optimizer,"deeplabv3plus_resnet50")
    test(deeplabv3plus_resnet50, test_loader, save_path="test_results_deeplabv3plus_resnet50",save_name="deeplabv3plus_resnet50")

if __name__ == "__main__":
    main()
