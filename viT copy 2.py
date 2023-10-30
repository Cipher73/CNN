import torch
import torch.nn as nn

from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import _load_weights
from sklearn.preprocessing import LabelEncoder
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import os
import glob


def resize_pos_embed(pos_embed, orig_grid_size, new_grid_size, num_extra_tokens):
    B, N, C = pos_embed.shape
    H, W = orig_grid_size
    new_H, new_W = new_grid_size

    if new_H == H and new_W == W:
        return pos_embed

    pos_embed = pos_embed[:, num_extra_tokens:]  # Remove extra tokens

    if new_H < H:
        pos_embed = pos_embed[:, :, :new_H, :]
    elif new_H > H:
        pad = torch.zeros(B, N, new_H - H, C, device=pos_embed.device)
        pos_embed = torch.cat([pos_embed, pad], dim=2)

    if new_W < W:
        pos_embed = pos_embed[:, :, :, :new_W]
    elif new_W > W:
        pad = torch.zeros(B, N, new_H, new_W - W, C, device=pos_embed.device)
        pos_embed = torch.cat([pos_embed, pad], dim=3)

    return pos_embed


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, channels):
        super(PatchEmbedding, self).__init__()

        self.image_size = image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image dimensions must be divisible by the patch size")
        self.grid_size = image_size[0] // patch_size, image_size[1] // patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, im):
        B, C, H, W = im.shape
        x = self.proj(im).flatten(2).transpose(1, 2)
        return x

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key=None, value=None, mask=None):
        batch_size = query.shape[0]

        # Linear projections for Q, K, and V
        q = self.wq(query)
        k = self.wk(key)
        v = self.wv(value)

        # Split into multiple heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled Dot-Product Attention
        attention_scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        output = torch.matmul(self.dropout(attention_weights), v)

        # Combine heads
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)

        # Linear projection
        output = self.fc_out(output)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, drop_path_rate):
  
        super(Block, self).__init__()
        
        # Multi-Head Self-Attention Layer
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-Forward Layer
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Multi-Head Self-Attention
        attn_output = self.self_attention(x)
        
        # Residual Connection and Layer Normalization
        x = x + attn_output
        x = self.norm1(x)
        
        # Feed-Forward Layer
        ff_output = self.feed_forward(x)
        
        # Residual Connection and Layer Normalization
        x = x + ff_output
        x = self.norm2(x)
        
        return x

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        # Initialize linear and convolutional layers with normal distribution
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        # Initialize LayerNorm with mean 1 and standard deviation 0.02
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0)

class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        n_layers,
        d_model,
        d_ff,
        n_heads,
        n_cls,
        dropout=0.1,
        drop_path_rate=0.0,
        distilled=False,
        channels=3,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size,
            patch_size,
            d_model,
            channels,
        )
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.n_cls = n_cls

        # cls and pos tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.distilled = distilled
        if self.distilled:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.patch_embed.num_patches + 2, d_model)
            )
            self.head_dist = nn.Linear(d_model, n_cls)
        else:
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.patch_embed.num_patches + 1, d_model)
            )

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        # output head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_cls)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        if self.distilled:
            trunc_normal_(self.dist_token, std=0.02)
        self.pre_logits = nn.Identity()

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward(self, im, return_features=False):
        B, _, H, W = im.shape
        PS = self.patch_size

        x = self.patch_embed(im)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.distilled:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self.pos_embed
        num_extra_tokens = 1 + self.distilled
        if x.shape[1] != pos_embed.shape[1]:
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (H // PS, W // PS),
                num_extra_tokens,
            )
        x = x + pos_embed
        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)
            print(x.shape,"x shape")
        x = self.norm(x)

        if return_features:
            return x

        if self.distilled:
            x, x_dist = x[:, 0], x[:, 1]
            x = self.head(x)
            x_dist = self.head_dist(x_dist)
            x = (x + x_dist) / 2
        else:
            x = x[:, 0]
            x = self.head(x)
        return x

    def get_attention_map(self, im, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
            )
        B, _, H, W = im.shape
        PS = self.patch_size

        x = self.patch_embed(im)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.distilled:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self.pos_embed
        num_extra_tokens = 1 + self.distilled
        if x.shape[1] != pos_embed.shape[1]:
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (H // PS, W // PS),
                num_extra_tokens,
            )
        x = x + pos_embed

        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)


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
        print(f"Number of classes = {n_classes}")
        y_train_cat = torch.tensor(y_train, dtype=torch.long)
        y_test_cat = torch.tensor(y_test, dtype=torch.long)
        y_val_cat = torch.tensor(y_val, dtype=torch.long)
        print(X_train[0].shape, y_train_cat[0].shape)
        #X_train = X_train.astype(np.float64)
        return X_train, y_train_cat, X_test, y_test_cat, X_val, y_val_cat, n_classes


import numpy as np
from torch.utils.data import DataLoader, random_split
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


image_size = (128, 128)  # Replace with your desired image size
patch_size = 16  # Replace with your desired patch size
n_layers = 12  # Replace with the number of layers you want
d_model = 768  # Replace with the desired model dimension
d_ff = 3072  # Replace with the desired feed-forward dimension
n_heads = 12  # Replace with the desired number of attention heads
n_cls = 11  # Replace with the number of output classes for your task

model = VisionTransformer(
    image_size=image_size,
    patch_size=patch_size,
    n_layers=n_layers,
    d_model=d_model,
    d_ff=d_ff,
    n_heads=n_heads,
    n_cls=n_cls,
)
# Define configuration parameters
learning_rate = 0.001
config = {
    'wandb_project': "SSOCTLRE",
    'num_epochs': 1,
    'batch_size': 16,
    'achitecture': ['Unet', 'DeepLabV3Plus'],
    'encoders': ['resnet34', 'resnet50'],
    'n_classes': n_classes,
    'activation': 'softmax',
    'learning_rate': learning_rate,
    'optimizer': "adam"
}

channels = 3  # Replace with the number of channels in your images

from torchsummary import summary

# Assuming you have already defined and instantiated your ViT model
model = VisionTransformer(image_size, patch_size, n_layers, d_model, d_ff, n_heads, n_cls)
model.to('cuda')  # Move the model to the GPU
dataloaders = {
    "train": DataLoader(list(zip(X_train, y_train_cat)), batch_size=config['batch_size'], shuffle=True),
    "valid": DataLoader(list(zip(X_val, y_val_cat)), batch_size=config['batch_size'], shuffle=False),
    "test": DataLoader(list(zip(X_test, y_test_cat)), batch_size=config['batch_size'], shuffle=False)
}

optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
criterion = nn.CrossEntropyLoss()

for epoch in range(config['num_epochs']):
    print(f"Epoch {epoch+1}/{config['num_epochs']}")
    print('-' * 10)
    for phase in ['train', 'valid']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to('cuda', dtype=torch.float)
            labels = labels.to('cuda', dtype=torch.float)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                print(inputs.shape)
                
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")