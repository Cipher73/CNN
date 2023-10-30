import torch
import torch.nn as nn
from vit_pytorch import ViT
import numpy as np
import cv2
import os
import glob
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score


class CustomViTForSegmentation(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0., emb_dropout=0.):
        super(CustomViTForSegmentation, self).__init__()
        self.vit = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
            channels=channels
        )

        # Update the number of output channels in the segmentation head
        self.segmentation_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling to get a 4D tensor
            nn.Conv2d(dim, num_classes, kernel_size=1)
        )

    def forward(self, x):
        print(x.shape,"x shape")
        x = self.vit(x)
        print(x.shape,"vit x shape")
        
        # Add spatial dimensions (height and width) to the tensor
        x = x.unsqueeze(-1).unsqueeze(-1)
        
        x = self.segmentation_head(x)  # Apply the segmentation head
        return x

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return running_loss / len(val_loader), accuracy

def test(model, test_loader, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())

    return all_preds

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
    
        
    
def main():
    data_processor = DataProcessor()
    image_size = 128
    patch_size = 32
    num_epochs = 5
    batch_size = 32
    learning_rate = 0.001
    dim = 1024
    depth = 6
    heads = 16
    mlp_dim = 2048

    

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
    model = CustomViTForSegmentation(image_size, patch_size, n_classes, dim, depth, heads, mlp_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create DataLoader objects for training, validation, and testing
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), y_train_cat)
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), y_val_cat)
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Testing
    test_preds = test(model, test_loader, device)
    # You can process the test_preds as needed for your task

if __name__ == "__main__":
    main()