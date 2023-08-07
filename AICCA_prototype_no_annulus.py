# Load Modules
import os
import numpy as np
import json
import math

# import seaborn as sns
# sns.set()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#from tqdm.notebook import tqdm

# Pytorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim


# cuda setup, set seed for reproducability 
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
set_seed(41)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print(f"Using device: {device}")


# YOU NEED THIS TO LOAD PyTorch Lightning I DON"T KNOW WHY

from jupyter_client.manager import KernelManager

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Set torch dtype to float64
torch.set_default_dtype(torch.float64)

# Modules for loading data, set data pathimport urllib
DATASET_PATH = "cloudimages1M" 
CHECKPOINT_PATH = os.getcwd()
DRIVE_PATH = "."

LOG_DIR = "ViT_1Mlr1e-5_no_annulus" 
training_checkpoint_path = f"/scratch/midway2/tdmonkman/AICCA_proj/ViT_1Mlr1e-5_no_annulus" 
import sys
sys.path.append(f"{DRIVE_PATH}/")

from import_npy import npy_loader


from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets

# Here we use our custom function imported from the drive directory (import_npy.py)
AICCA_data = datasets.DatasetFolder(root=f"{DRIVE_PATH}/{DATASET_PATH}",
                                    loader=npy_loader,
                                    extensions=tuple('.npy'))
    

print(AICCA_data)
print("classes")
print(AICCA_data.classes)
print("class_dict")
print(AICCA_data.class_to_idx)

# Data loader works her
# Split into train, validation, and test data
train_length = int(0.7*len(AICCA_data))
validation_length = int(0.2*len(AICCA_data))
test_length = len(AICCA_data) - train_length - validation_length

# Split
train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(AICCA_data, (train_length, validation_length, test_length))

# Verify size of datasets
print(f"size train_dataset: {len(train_dataset)}")
print(f"size validation_dataset: {len(validation_dataset)}")
print(f"size test_dataset: {len(test_dataset)}")

# 
train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=False, pin_memory=True)
val_loader = data.DataLoader(validation_dataset, batch_size=128, shuffle=False, drop_last=False, pin_memory=True)
test_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False, pin_memory=True)





########################################################################################################
########################################################################################################
# Function for preprocessing the images into patches, cutting out the annulus
def imgs_to_patches_noannulus(imgs, patch_size, flatten_channels=True):
    """
    Inputs
    ------
    imgs: torch.Tensor containing the images of shape (Num Images, Channels, Height, Width)
    patch_size: 
    flatten_channels: False
    
    Outputs
    -------
    imgs: 
    """
    # Reshape the image tensor to shape (image, channel, height, width) 
    imgs = imgs.permute(0,3,1,2)[:,:,19:109,19:109]
    B, C, H, W = imgs.shape
    # Divide images into patches
    imgs = imgs.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    # Reshape images to shape (image, patch, channel, height, width)
    imgs = imgs.permute(0,2,4,1,3,5)
    imgs = imgs.flatten(1,2)
    # You can flatten the patches into a "feature vector" if you would like
    if flatten_channels:
        imgs = imgs.flatten(2,4)
    return imgs
    

########################################################################################################
########################################################################################################
class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs
        ------
        embed_dim: Dimensionality of input and attention feature vectors
        hidden_dim: Dimensionality of hidden layer in feed-forward network
                    (usually 2 to 4x larger than embed_dim)
        num_heads: Number of heads to use in the Multi-Head Attention block
        dropout: Amount of dropout to apply in the feed-forward network
        """
        super().__init__() # super() is a function that allows you to initialize 
                           # attributes from the parent class
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        # nn.LayerNorm applies layer normalization over a mini-batch of inputs as
        # described in the AttentionBlock paper. Uses the expectation value and 
        # mean to calculate them over the last D-dimensions where D is the 
        # dimension of the 'normalized_shape.'
        
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        # nn.MultiheadAttention applies the multiheaded attention layer from the
        # Attention paper. 'embed_dim' and 'num_heads' are self-explanatory.
        # During training, dropout randomly zeros some of the elements of the 
        # input tensor with probability p using samples from a Bernoulli 
        # distribution. Each channel will be zeroed out independently on
        # every forward call. This is an effective technique for regularization
        # and preventing the co-adaption of neurons.
        
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        # take the norm again
        
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        # run the attention block
        input_x = self.layer_norm_1(x)
        x = x + self.attn(input_x, input_x, input_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x
    
    
########################################################################################################
########################################################################################################
class VisionTransformer(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches, dropout=0.0):
        """
        Inputs
        ------
        embed_dim: Dimensionality of the input feature vectors to the Transformer
        hidden_dim: Dimensionality of the hidden layer in the feed-forward networks
                     within the Transformer
        num_channels: Number of channels of the input (3 for RGB)
        num_heads: Number of heads to use in the Multi-Head Attention block
        num_layers: Number of layers to use in the Transformer
        num_classes: Number of classes to predict
        patch_size: Number of pixels that the patches have per dimension
        num_patches: Maximum number of patches an image can have
        dropout: Amount of dropout to apply in the feed-forward network and
                  on the input encoding
        
        Dependencies
        ------------
        
        """
        super().__init__()
        
        self.patch_size = patch_size
        
        # Layers/Networks
        self.input_layer = nn.Linear(num_channels*(patch_size**2), embed_dim)
        self.transformer = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for layer in range(num_layers)])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)
        
        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,1+num_patches,embed_dim))
        
    def forward(self, img):
        # Preprocess input
        img = imgs_to_patches_noannulus(img, self.patch_size)
        B, T, _ = img.shape
        img = self.input_layer(img)
        
        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        img = torch.cat([cls_token, img], dim=1)
        img = img + self.pos_embedding[:,:T+1]
        
        # Apply Transformer
        img = self.dropout(img)
        img = img.transpose(0,1)
        img = self.transformer(img)
        
        # Perform classification prediction
        cls = img[0]
        out = self.mlp_head(cls)
        return out
    
########################################################################################################
########################################################################################################
from lightning_fabric.utilities import optimizer

class ViT(pl.LightningModule):

    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = VisionTransformer(**model_kwargs)
        self.example_input_array = next(iter(train_loader))[0]

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(f'{mode}_loss', loss)
        self.log(f'{mode}_acc', acc)
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")


def train_model(training_checkpoint_path, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, LOG_DIR),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=180,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    LearningRateMonitor("epoch")])
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "last.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = ViT.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
    
    elif os.path.isfile(training_checkpoint_path):
        print(f"Found checkpoint at {training_checkpoint_path}")
        model = ViT.load_from_checkpoint(training_checkpoint_path) # Load best checkpoint after training
        trainer.fit(model, train_loader, val_loader)
        model = ViT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    else:
        print("Running from scratch")
        #pl.seed_everything(42) # To be reproducable
        model = ViT(**kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = ViT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    return model, result

########################################################################################################
########################################################################################################



model, results =     train_model( training_checkpoint_path, model_kwargs={
                                    'embed_dim': 180,
                                    'hidden_dim': 360,
                                    'num_heads': 6,
                                    'num_layers': 6,
                                    'patch_size': 15,
                                    'num_channels': 6,
                                    'num_patches': 36,
                                    'num_classes': 43,
                                    'dropout': 0.2
 			 }, lr=1e-05,) #LEARNING RATE FLAG