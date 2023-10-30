import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.datasets import DatasetFolder
from PIL import Image
import os

# Set up your data paths
source_data_path = 'Dataset/train/noisy_images'
target_data_path = 'Dataset/train/clean_images'

class CustomDataset(Dataset):
    def __init__(self, root, transform=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.image_paths = sorted(os.listdir(self.root))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.image_paths[idx])
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image

# Define your data transformations
data_transforms = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((128, 128)),  # Keep the size as (128, 128)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Create custom datasets for source and target domains
source_dataset = CustomDataset(root=source_data_path, transform=data_transforms)
target_dataset = CustomDataset(root=target_data_path, transform=data_transforms)

# Create data loaders
source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True, num_workers=4)
target_dataloader = DataLoader(target_dataset, batch_size=32, shuffle=True, num_workers=4)

# Define your model architecture
class FeatureExtractor(nn.Module):
    # Your FeatureExtractor class

class Classifier(nn.Module):
    # Your Classifier class

class DomainClassifier(nn.Module):
    # Your DomainClassifier class

# Set up your model and optimizers
feature_extractor = FeatureExtractor()
classifier = Classifier()
domain_classifier = DomainClassifier()

optimizer_f = optim.Adam(feature_extractor.parameters())
optimizer_c = optim.Adam(classifier.parameters())
optimizer_d = optim.Adam(domain_classifier.parameters())

# Define your loss functions
criterion_classification = nn.CrossEntropyLoss()
criterion_domain = nn.BCELoss()  # Binary cross-entropy loss for domain classification

hidden_dims = 512

# Training loop with domain adaptation
def domain_adaptation(feature_extractor, classifier, domain_classifier,
                      source_dataloader, target_dataloader,
                      optimizer_f, optimizer_c, optimizer_d, criterion_classification, criterion_domain, num_epochs):
    for epoch in range(num_epochs):
        for source_data, target_data in zip(source_dataloader, target_dataloader):
            source_images = source_data
            target_images = target_data

            # Zero the gradients
            optimizer_f.zero_grad()
            optimizer_c.zero_grad()
            optimizer_d.zero_grad()

            # Feature extraction
            source_features = feature_extractor(source_images)
            target_features = feature_extractor(target_images)

            # Classification loss on source domain (if source_labels are available)
            source_labels = None  # You need to define source_labels if available
            if source_labels is not None:
                source_classifier_output = classifier(source_features)
                classification_loss = criterion_classification(source_classifier_output, source_labels)
            else:
                classification_loss = 0.0  # No source labels available

            # Domain classification loss
            source_domain_labels = torch.zeros(source_features.size(0), dtype=torch.float32)  # Source domain label is 0
            target_domain_labels = torch.ones(target_features.size(0), dtype=torch.float32)  # Target domain label is 1
            domain_labels = torch.cat((source_domain_labels, target_domain_labels), dim=0)

            source_features, target_features = source_features.view(-1, hidden_dims), target_features.view(-1, hidden_dims)
            all_features = torch.cat((source_features, target_features), dim=0)

            domain_classifier_output = domain_classifier(all_features)
            domain_loss = criterion_domain(domain_classifier_output, domain_labels)

            # Total loss
            total_loss = classification_loss + domain_loss

            # Backpropagation and optimization
            total_loss.backward()
            optimizer_f.step()
            optimizer_c.step()
            optimizer_d.step()

            # Print loss for monitoring
            print(f"Epoch {epoch + 1}, Total Loss: {total_loss.item()}, Classification Loss: {classification_loss.item()}, Domain Loss: {domain_loss.item()}")

# Train your model with domain adaptation
num_epochs = 10
domain_adaptation(feature_extractor, classifier, domain_classifier, source_dataloader, target_dataloader, optimizer_f, optimizer_c, optimizer_d, criterion_classification, criterion_domain, num_epochs)
