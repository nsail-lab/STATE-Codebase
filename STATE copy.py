#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import sys, json

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# In[2]:


# In[2]:

#####################################
# Hyperparameters and Configurations #
#####################################
CUDA = True
# Paths (Update these paths as per your setup)
root_directory = 'dataset/output_padding_0.4_size_128'  # Root directory containing trajectory subdirectories
json_annotations = 'dataset/20240715_dtm_annotations.csv.json'  # Path to your JSON file

BATCH_SIZE = 64  # Adjusted for example
IMAGE_CHANNEL = 1
Z_DIM = 100
CLIP_NUM_VECTORS = 2
CLIP_VECTOR_DIM = 512
CLIP_DIM = CLIP_NUM_VECTORS * CLIP_VECTOR_DIM  # 2 * 512 = 1024
LABEL_DIM = 2
G_HIDDEN = 64
D_HIDDEN = 64
X_DIM = 128   # Image size 128x128
EPOCH_NUM = 2000
REAL_LABEL = 1
FAKE_LABEL = 0
lr = 2e-4
seed = 1


# In[3]:


# In[3]:

#####################################
# Device setup
#####################################
CUDA = CUDA and torch.cuda.is_available()
print("PyTorch version: {}".format(torch.__version__))
if CUDA:
    print("CUDA version: {}\n".format(torch.version.cuda))

if CUDA:
    torch.cuda.manual_seed(seed)
device = torch.device("cuda:0" if CUDA else "cpu")
cudnn.benchmark = True

class CombinedBinaryClassifier(nn.Module):
    """
    - Flattens the 128×128 binary trajectory image and maps it to hidden_dim.
    - Processes each of the 5 CLIP vectors (512→hidden_dim).
    - Concatenates all six hidden_dim vectors, fuses to fusion_dim, then outputs logits.
    """
    def __init__(self,
                 clip_feature_dim: int = 512,
                 img_size: int = 128,
                 hidden_dim: int = 256,
                 fusion_dim: int = 512,
                 num_classes: int = 1,
                 dropout: float = 0.5):
        super().__init__()
        # --- simple image branch: flatten + single FC ---
        self.img_processor = nn.Sequential(
            nn.Flatten(),                              # [B, 1*128*128]
            nn.Linear(img_size * img_size, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(dropout)
        )

        # --- CLIP feature branches ---
        self.feature_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(clip_feature_dim, hidden_dim),
                nn.ReLU(True),
                nn.Dropout(dropout)
            )
            for _ in range(5)
        ])

        # --- fusion & classification ---
        total_feats = hidden_dim * 6  # 5 CLIP + 1 image
        self.fusion = nn.Sequential(
            nn.Linear(total_feats, fusion_dim),
            nn.ReLU(True),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, clip_feats: torch.Tensor, traj_img: torch.Tensor) -> torch.Tensor:
        """
        clip_feats: [B, C, 512]  # now C==2
        traj_img:   [B, 1, 128, 128]
        """
        B, C, D = clip_feats.shape
        num_procs = len(self.feature_processors)  # ==5

        # ─── If we got fewer than 5 CLIP vectors, pad with zeros ───
        if C < num_procs:
            pad = torch.zeros(B, num_procs - C, D,
                              device=clip_feats.device,
                              dtype=clip_feats.dtype)
            clip_feats = torch.cat([clip_feats, pad], dim=1)
        # ─────────────────────────────────────────────────────────────

        out_feats = []
        for i, proc in enumerate(self.feature_processors):
            out_feats.append(proc(clip_feats[:, i, :]))  # now safe for i=0..4

        img_feat = self.img_processor(traj_img)           # [B, hidden_dim]
        out_feats.append(img_feat)

        x = torch.cat(out_feats, dim=1)
        x = self.fusion(x)
        logits = self.classifier(x)
        return logits.view(-1)

# In[4]:


# Load the frozen threat‐alignment network
DT_MODEL_PATH = '/home/td/Desktop/TrajectoryGeneration/Asset-Value-Threat-Classifier/cv_models_v3/fold_1_best.pth'
netT = CombinedBinaryClassifier().to(device)
netT.load_state_dict(torch.load(DT_MODEL_PATH, map_location=device))
netT.eval()
for p in netT.parameters():
    p.requires_grad = False
    
# weighting coefficients from the paper
lambda_V = 0.6                                                     # ◀◀ validity weight
lambda_T = 0.4                                                     # ◀◀ threat‐alignment weight

# we’ll use a second BCELoss for the classification term
criterion_cls = nn.BCELoss()


# In[ ]:


# In[4]:

#####################################
# Dataset Definition
#####################################
class TrajectoryDataset(Dataset):
    def __init__(self, root, json_file, threshold=6, transform=None):
        """
        Args:
            root (str): Root directory containing subdirectories for each trajectory_id.
                        Each subdirectory must contain:
                          - {trajectory_id}_trajectory.png
                          - {trajectory_id}_CLIP.npy
            json_file (str): Path to the JSON file containing annotations.
            threshold (int): Threshold for binarizing the overall_threat_score.
            transform (callable, optional): A transform to apply to the trajectory image.
        """
        self.root = root
        self.transform = transform
        self.threshold = threshold
        self.binarized_scores = self._prepare_binarized_scores(json_file)
        self.samples = self._collect_samples()

    def _prepare_binarized_scores(self, json_file):
        """
        Loads the JSON file and prepares a dictionary mapping trajectory_id to binarized threat score.

        Args:
            json_file (str): Path to the JSON file.

        Returns:
            dict: Mapping of trajectory_id to binarized score (0 or 1).
        """
        try:
            with open(json_file, 'r') as f:
                annotations = json.load(f)
            logging.info(f"Loaded {len(annotations)} annotations from {json_file}.")
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON: {e}")
            raise
        except FileNotFoundError:
            logging.error(f"JSON file not found: {json_file}")
            raise

        threat_scores = {}
        for annotation in annotations:
            trajectory_id = str(annotation.get("trajectory_id"))
            score = annotation.get("overall_threat_score", 0)

            if trajectory_id in threat_scores:
                if score > threat_scores[trajectory_id]:
                    threat_scores[trajectory_id] = score
            else:
                threat_scores[trajectory_id] = score

        logging.info(f"Extracted max threat scores for {len(threat_scores)} unique trajectory IDs.")

        # Binarize the scores based on the threshold
        binarized_scores = {traj_id: (1 if score >= self.threshold else 0) 
                            for traj_id, score in threat_scores.items()}
        logging.info(f"Binarized threat scores using threshold {self.threshold}.")

        return binarized_scores

    def _collect_samples(self):
        """
        Collects all valid samples from the root directory.

        Returns:
            list: List of tuples containing (trajectory_id, trajectory_dir).
        """
        samples = []
        for traj_id in os.listdir(self.root):
            traj_dir = os.path.join(self.root, traj_id)
            if os.path.isdir(traj_dir):
                traj_img = os.path.join(traj_dir, f"{traj_id}_trajectory.png")
                traj_clip = os.path.join(traj_dir, f"{traj_id}_CLIP.npy")
                if os.path.isfile(traj_img) and os.path.isfile(traj_clip):
                    if traj_id in self.binarized_scores:
                        samples.append((traj_id, traj_dir))
                    else:
                        logging.warning(f"No binarized score for trajectory ID {traj_id}. Skipping.")
                else:
                    logging.warning(f"Missing files in {traj_dir}. Expected .png and _CLIP.npy files.")
        logging.info(f"Collected {len(samples)} valid samples.")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Retrieves the trajectory image, CLIP feature array, and one-hot encoded label.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: (trajectory_image, clip_features, one_hot_label)
                   - trajectory_image (torch.Tensor): Transformed trajectory image.
                   - clip_features (torch.Tensor): Tensor of shape [2, 512].
                   - one_hot_label (torch.Tensor): One-hot encoded label (2 classes).
        """
        trajectory_id, traj_dir = self.samples[index]

        # Load trajectory image
        traj_img_path = os.path.join(traj_dir, f"{trajectory_id}_trajectory.png")
        try:
            img = Image.open(traj_img_path).convert('L')  # Convert to grayscale
        except Exception as e:
            logging.error(f"Error loading image {traj_img_path}: {e}")
            raise

        if self.transform:
            img = self.transform(img)
        else:
            # Default transformation: Convert to tensor
            img = transforms.ToTensor()(img)

        # Load the CLIP numpy array
        traj_clip_path = os.path.join(traj_dir, f"{trajectory_id}_CLIP.npy")
        try:
            clip_features = np.load(traj_clip_path)  # Expected shape: (5, 512)
            # take the first two vectors
            clip_features = clip_features[:2]  # Shape: [2, 512]
        except Exception as e:
            logging.error(f"Error loading CLIP numpy file {traj_clip_path}: {e}")
            raise

        if clip_features.ndim != 2 or clip_features.shape != (CLIP_NUM_VECTORS, CLIP_VECTOR_DIM):
            logging.error(f"Expected CLIP features of shape ({CLIP_NUM_VECTORS}, {CLIP_VECTOR_DIM}) in {traj_clip_path}, got {clip_features.shape}.")
            raise ValueError(f"Invalid CLIP feature shape for {traj_clip_path}.")

        # Convert to torch tensor
        clip_features_tensor = torch.tensor(clip_features, dtype=torch.float32)  # Shape: [2, 512]

        # Get the binary threat label and convert to one-hot
        label_binary = self.binarized_scores[trajectory_id]
        one_hot_label = torch.zeros(LABEL_DIM, dtype=torch.float32)
        one_hot_label[label_binary] = 1.0

        return img, clip_features_tensor, one_hot_label


# In[6]:


# In[5]:

# Initialize the dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize((X_DIM, X_DIM)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Initialize the dataset
dataset = TrajectoryDataset(
    root=root_directory,
    json_file=json_annotations,
    threshold=3,  # Set your desired threshold
    # In the DEWS paper we had three schemes: LTP (scores greater than 3 was threatening), MTP (scores greater than 6 was threatening), 
    # HTP (scores greater than 8 was threatening). In all cases, the classification was binary
    transform=transform
)

# Initialize the DataLoader
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
logging.info("DataLoader initialized.")

# Quick sanity check
images, clip_features, labels = next(iter(dataloader))
print(f" - Images shape: {images.shape}")                # e.g., [64, 1, 128, 128]
print(f" - CLIP Features shape: {clip_features.shape}")  # e.g., [64, 2, 512]
print(f" - Labels shape: {labels.shape}")                # e.g., [64, 2]

#####################################
# Visualization of training images
#####################################
# Uncomment the following lines to visualize a batch of training images
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(images.to(device)[:16], padding=2, normalize=True).cpu(),(1,2,0)))
# plt.show()


# In[7]:


# In[6]:

#####################################
# Weight initialization
#####################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# In[8]:


# In[7]:

#####################################
# Generator and Discriminator Definitions
#####################################

class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM, clip_num_vectors=CLIP_NUM_VECTORS, clip_vector_dim=CLIP_VECTOR_DIM, label_dim=LABEL_DIM, hidden_dim=G_HIDDEN, image_channel=IMAGE_CHANNEL):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.clip_num_vectors = clip_num_vectors
        self.clip_vector_dim = clip_vector_dim
        self.clip_dim = clip_num_vectors * clip_vector_dim
        self.label_dim = label_dim
        self.hidden_dim = hidden_dim
        self.image_channel = image_channel

        # Process noise
        self.noise_fc = nn.Sequential(
            nn.Linear(z_dim, hidden_dim * 8 * 4 * 4),
            nn.BatchNorm1d(hidden_dim * 8 * 4 * 4),
            nn.ReLU(True)
        )

        # Process each CLIP vector independently
        self.clip_fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(clip_vector_dim, hidden_dim * 8 * 4 * 4),
                nn.BatchNorm1d(hidden_dim * 8 * 4 * 4),
                nn.ReLU(True)
            ) for _ in range(clip_num_vectors)
        ])

        # Process labels
        self.label_fc = nn.Sequential(
            nn.Linear(label_dim, hidden_dim * 8 * 4 * 4),
            nn.BatchNorm1d(hidden_dim * 8 * 4 * 4),
            nn.ReLU(True)
        )

        # Combine all processed features
        # Total features after concatenation: noise + 2 CLIP + labels
        self.combine_fc = nn.Sequential(
            nn.Linear(hidden_dim * 8 * 4 * 4 * (1 + clip_num_vectors + 1), hidden_dim * 8 * 4 * 4),
            nn.BatchNorm1d(hidden_dim * 8 * 4 * 4),
            nn.ReLU(True)
        )

        # Main generator network
        self.main = nn.Sequential(
            # Reshape to (N, G_HIDDEN*8, 4, 4)
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, 4, 2, 1, bias=False),  # 4x4 -> 8x8
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, bias=False),  # 8x8 -> 16x16
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1, bias=False),      # 16x16 -> 32x32
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, 2, 1, bias=False),    # 32x32 -> 64x64
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_dim // 2, image_channel, 4, 2, 1, bias=False),# 64x64 -> 128x128
            nn.Tanh()
        )

    def forward(self, noise, clip_features, labels):
        # noise: [B, Z_DIM]
        # clip_features: [B, 2, 512]
        # labels: [B, LABEL_DIM]
        B = noise.size(0)

        # Process noise
        noise_processed = self.noise_fc(noise)  # [B, hidden_dim*8*4*4]

        # Process each CLIP vector independently
        clip_processed = []
        for i in range(self.clip_num_vectors):
            clip_vector = clip_features[:, i, :]  # [B, 512]
            processed = self.clip_fcs[i](clip_vector)  # [B, hidden_dim*8*4*4]
            clip_processed.append(processed)
        clip_processed = torch.cat(clip_processed, dim=1)  # [B, hidden_dim*8*4*4 * 2]

        # Process labels
        label_processed = self.label_fc(labels)  # [B, hidden_dim*8*4*4]

        # Combine all processed features
        combined = torch.cat([noise_processed, clip_processed, label_processed], dim=1)  # [B, hidden_dim*8*4*4 * (1 + 2 +1)]

        # Further process combined features
        combined = self.combine_fc(combined)  # [B, hidden_dim*8*4*4]

        # Reshape to [B, hidden_dim*8, 4, 4]
        combined = combined.view(-1, self.hidden_dim * 8, 4, 4)

        # Pass through main generator layers
        output = self.main(combined)  # [B, IMAGE_CHANNEL, 128, 128]

        return output  # [B, 1, 128, 128]


class Discriminator(nn.Module):
    def __init__(self, clip_num_vectors=CLIP_NUM_VECTORS, clip_vector_dim=CLIP_VECTOR_DIM, label_dim=LABEL_DIM, hidden_dim=D_HIDDEN, image_channel=IMAGE_CHANNEL, image_size=X_DIM):
        super(Discriminator, self).__init__()
        self.clip_num_vectors = clip_num_vectors
        self.clip_vector_dim = clip_vector_dim
        self.clip_dim = clip_num_vectors * clip_vector_dim
        self.label_dim = label_dim
        self.hidden_dim = hidden_dim
        self.image_channel = image_channel
        self.image_size = image_size

        # Define the CLIP compression layers
        self.clip_compression = nn.Sequential(
            nn.Linear(CLIP_DIM, hidden_dim * 2),  # Compress from 1024 to 128 (assuming hidden_dim=64)
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim * 2, hidden_dim),  # Compress to 64
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True)
        )

        # Define the label projection layer
        self.label_proj = nn.Sequential(
            nn.Linear(label_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Define the convolutional layers
        # Input channels: 1 (image) + hidden_dim (compressed CLIP) + hidden_dim (labels) = 1 + 64 + 64 = 129
        self.main = nn.Sequential(
            nn.Conv2d(1 + hidden_dim + hidden_dim, D_HIDDEN, 4, 2, 1, bias=False),  # [B, D_HIDDEN, 64, 64]
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, 4, 2, 1, bias=False),  # [B, D_HIDDEN*2, 32, 32]
            nn.BatchNorm2d(D_HIDDEN * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, 4, 2, 1, bias=False),  # [B, D_HIDDEN*4, 16, 16]
            nn.BatchNorm2d(D_HIDDEN * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, 4, 2, 1, bias=False),  # [B, D_HIDDEN*8, 8, 8]
            nn.BatchNorm2d(D_HIDDEN * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Final layer to reduce to [B,1,1,1]
            nn.Conv2d(D_HIDDEN * 8, 1, 8, 8, 0, bias=False),          # [B,1,1,1]
            nn.Sigmoid()
        )
    
    def forward(self, img, clip_features, labels):
        # img: [B,1,128,128]
        # clip_features: [B,2,512]
        # labels: [B,2]
        B = img.size(0)

        # Flatten CLIP features
        clip_features_flat = clip_features.view(B, -1)  # [B, 1024]

        # Compress CLIP features
        clip_compressed = self.clip_compression(clip_features_flat)  # [B, 64]

        # Process labels
        label_processed = self.label_proj(labels)  # [B, 64]

        # Expand compressed CLIP features and labels to match image spatial dimensions
        clip_expanded = clip_compressed.unsqueeze(-1).unsqueeze(-1)  # [B, 64, 1, 1]
        clip_expanded = clip_expanded.repeat(1, 1, self.image_size, self.image_size)  # [B, 64, 128, 128]

        label_expanded = label_processed.unsqueeze(-1).unsqueeze(-1)      # [B, 64, 1, 1]
        label_expanded = label_expanded.repeat(1, 1, self.image_size, self.image_size)  # [B, 64, 128, 128]

        # Concatenate image, compressed CLIP features, and labels
        x = torch.cat([img, clip_expanded, label_expanded], dim=1)   # [B, 1 + 64 + 64, 128, 128]

        # Pass through convolutional layers
        output = self.main(x)  # [B,1,1,1]
        output = output.view(-1)  # [B]

        return output  # [B]


# In[9]:


# In[8]:

#####################################
# Initialize networks
#####################################
# Initialize Generator
netG = Generator().to(device)
netG.apply(weights_init)
print(netG)

# Initialize Discriminator
netD = Discriminator().to(device)
netD.apply(weights_init)
print(netD)


# In[10]:


# In[9]:

#####################################
# Loss and Optimizer
#####################################
criterion = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))


# In[11]:


# In[10]:

#####################################
# Visualization Setup
#####################################
# Function to get a fixed batch for visualization
def get_fixed_batch(dataloader, device, batch_size):
    for data in dataloader:
        images, clip_features, labels = data
        images = images.to(device)
        clip_features = clip_features.to(device)
        labels = labels.to(device)
        return images[:batch_size], clip_features[:batch_size], labels[:batch_size]

# Retrieve a fixed batch
fixed_real_images, fixed_real_clip, fixed_real_labels = get_fixed_batch(dataloader, device, BATCH_SIZE)

# Sample fixed noise
fixed_noise = torch.randn(BATCH_SIZE, Z_DIM, device=device)

#####################################
# Interactive Visualization Function
#####################################
def visualize_by_index(fake_images_list, fake_labels_list, fixed_real_images, fixed_real_labels, index, images_per_row=8):
    """
    Visualize fake images based on a specific index.

    Each row displays up to 'images_per_row' images:
    - Top row: Generated (Fake) images

    All images in a single row belong to the same label.

    Parameters:
    - fake_images_list: List of tensors containing fake images.
    - fake_labels_list: List of tensors containing labels for fake images.
    - fixed_real_images: Tensor of real images.
    - fixed_real_labels: Tensor of labels for real images.
    - index: The index of the set to visualize.
    - images_per_row: Number of images per row (max 8 as per requirement).
    """

    if index < 0 or index >= len(fake_images_list):
        print("Index out of range. Please provide a valid index.")
        return

    # Retrieve the fake images and labels at the specified index
    fake_images = fake_images_list[index]
    fake_labels = fake_labels_list[index]

    # Convert labels to integer indices
    fake_labels = torch.argmax(fake_labels, dim=1).numpy()

    # Get unique labels in this set
    unique_labels = np.unique(fake_labels)

    for label in unique_labels:
        # Find indices for the current label
        fake_indices = np.where(fake_labels == label)[0]

        # Select up to 'images_per_row' images
        selected_fake = fake_indices[:images_per_row]

        # If less than 'images_per_row' images are available, pad with available ones
        num_fake = len(selected_fake)

        if num_fake < images_per_row:
            print(f"Label {label}: Only {num_fake} fake images available for visualization.")

        # Extract selected images
        fake_imgs = fake_images[selected_fake]

        # Create a figure with one row: fake images
        fig, axes = plt.subplots(1, images_per_row, figsize=(images_per_row * 2, 2))
        fig.suptitle(f'Label {label} - Set {index}', fontsize=16)

        for i in range(images_per_row):
            if i < num_fake:
                ax = axes[i]
                img_fake = fake_imgs[i].squeeze().numpy()
                ax.imshow(img_fake, cmap='gray')
                ax.axis('off')
            else:
                axes[i].axis('off')

        plt.tight_layout()
        plt.show()


# In[12]:


# In[11]:
#####################################
# Training Loop with Model Saving
#####################################
# Initialize lists to store fake images and their labels
fake_images_list = []
fake_labels_list = []
G_losses = []
D_losses = []
iters = 0

# Directories for saving models and images
model_save_dir = 'saved_models_0.4_DT'
image_save_dir = 'generated_images_0.4_DT'

# Create directories if they don't exist
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(image_save_dir, exist_ok=True)

print("Starting Training Loop...")
for epoch in range(EPOCH_NUM):
    for i, (images, clip_features, labels) in enumerate(dataloader, 0):
        images = images.to(device)                     # [B,1,128,128]
        clip_features = clip_features.to(device)       # [B,2,512]
        labels = labels.to(device)                     # [B,2]
        b_size = images.size(0)

        ############################################################
        # (1) Update D: maximize log(D(x)) + log(1 - D(G(z,h,l)))  #
        ############################################################        
        netD.zero_grad()
        # Train with real data
        real_label_tensor = torch.full((b_size,), REAL_LABEL, dtype=torch.float, device=device)
        output_real = netD(images, clip_features, labels)  # [B]
        # Ensure output is [B]
        assert output_real.shape == real_label_tensor.shape, f"Discriminator output shape {output_real.shape} does not match real label shape {real_label_tensor.shape}"
        errD_real = criterion(output_real, real_label_tensor)
        errD_real.backward()
        D_x = output_real.mean().item()

        # Train with fake data
        noise = torch.randn(b_size, Z_DIM, device=device)
        fake_images = netG(noise, clip_features, labels)      # [B,1,128,128]
        fake_label_tensor = torch.full((b_size,), FAKE_LABEL, dtype=torch.float, device=device)
        output_fake = netD(fake_images.detach(), clip_features, labels)  # [B]
        # Ensure output is [B]
        assert output_fake.shape == fake_label_tensor.shape, f"Discriminator output shape {output_fake.shape} does not match fake label shape {fake_label_tensor.shape}"
        errD_fake = criterion(output_fake, fake_label_tensor)
        errD_fake.backward()
        D_G_z1 = output_fake.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################################################
        # (2) Update G: maximize log(D(G(z,h,l), h, l))                  #
        ############################################################
        netG.zero_grad()
        # Since we want to fool D, use real_label_tensor for fake inputs
        output_fake_for_G = netD(fake_images, clip_features, labels)  # [B]
        # Ensure output is [B]
        assert output_fake_for_G.shape == real_label_tensor.shape, f"Discriminator output shape {output_fake_for_G.shape} does not match real label shape {real_label_tensor.shape}"
        adv_loss = criterion(output_fake_for_G, real_label_tensor)
        D_G_z2 = output_fake_for_G.mean().item()

        # 2b) — Threat‐alignment loss via frozen netT
        #    CombinedBinaryClassifier returns a single logit per sample
        cls_logits = netT(clip_features, fake_images)                     # [B,1] or [B]
        cls_prob   = torch.sigmoid(cls_logits).view(-1)                   # [B]
        # extract the 0/1 target from your one‐hot labels
        target_scalar = labels[:,1]                                       # [B], 1 for “threatening”, 0 otherwise
        cls_loss = criterion_cls(cls_prob, target_scalar)

        # 2c) — Combined generator loss
        errG = lambda_V * adv_loss + lambda_T * cls_loss
        errG.backward()
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z,h,l)): %.4f / %.4f'
                  % (epoch+1, EPOCH_NUM, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save losses
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check generator progress and store fake images
        if (iters % 500 == 0) or ((epoch == EPOCH_NUM-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                # Generate fake images using fixed noise and fixed real CLIP features and labels
                fake_images_fixed = netG(fixed_noise, fixed_real_clip, fixed_real_labels).detach().cpu()
            # Store fake images and corresponding labels
            fake_images_list.append(fake_images_fixed)
            fake_labels_list.append(fixed_real_labels.cpu())

            # Save generated images to disk
            img_grid = vutils.make_grid(fake_images_fixed, padding=2, normalize=True)
            img_save_path = os.path.join(image_save_dir, f'epoch_{epoch+1}_iter_{i}.png')
            vutils.save_image(fake_images_fixed, img_save_path, normalize=True)
            logging.info(f"Saved generated images to {img_save_path}")

            # Save model checkpoints
            # model_save_path_G = os.path.join(model_save_dir, f'generator_epoch_{epoch+1}_iter_{i}.pth')
            # model_save_path_D = os.path.join(model_save_dir, f'discriminator_epoch_{epoch+1}_iter_{i}.pth')
            # torch.save(netG.state_dict(), model_save_path_G)
            # torch.save(netD.state_dict(), model_save_path_D)
            # logging.info(f"Saved Generator to {model_save_path_G}")
            # logging.info(f"Saved Discriminator to {model_save_path_D}")

        iters += 1
    # Optional: Save models at the end of training
    # if (epoch == EPOCH_NUM-1):
    #     model_save_path_G = os.path.join(model_save_dir, f'generator_final_epoch_{epoch+1}.pth')
    #     model_save_path_D = os.path.join(model_save_dir, f'discriminator_final_epoch_{epoch+1}.pth')
    #     torch.save(netG.state_dict(), model_save_path_G)
    #     torch.save(netD.state_dict(), model_save_path_D)
    #     logging.info(f"Saved final Generator to {model_save_path_G}")
    #     logging.info(f"Saved final Discriminator to {model_save_path_D}")
    
    # save every 50 epochs
    if (epoch+1) % 50 == 0:
        model_save_path_G = os.path.join(model_save_dir, f'generator_epoch_{epoch+1}.pth')
        torch.save(netG.state_dict(), model_save_path_G)
        logging.info(f"Saved Generator to {model_save_path_G}")


# In[15]:


clip_features.shape


# In[12]:


# In[12]:

#####################################
# Plotting Losses
#####################################
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="Generator")
plt.plot(D_losses,label="Discriminator")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[13]:


# In[13]:
#####################################
# Interactive Visualization Function
#####################################
def visualize_by_index(fake_images_list, fake_labels_list, fixed_real_images, fixed_real_labels, index, images_per_row=8):
    """
    Visualize real and fake images based on a specific index.

    Each row displays up to 'images_per_row' images:
    - Top row: Real images
    - Bottom row: Generated (Fake) images

    All images in a single row belong to the same label.

    Parameters:
    - fake_images_list: List of tensors containing fake images.
    - fake_labels_list: List of tensors containing labels for fake images.
    - fixed_real_images: Tensor of real images.
    - fixed_real_labels: Tensor of labels for real images.
    - index: The index of the set to visualize.
    - images_per_row: Number of images per row (max 8 as per requirement).
    """

    if index < 0 or index >= len(fake_images_list):
        print("Index out of range. Please provide a valid index.")
        return

    # Retrieve the fake images and labels at the specified index
    fake_images = fake_images_list[index]
    fake_labels = fake_labels_list[index]

    # Convert labels to integer indices
    fake_labels = torch.argmax(fake_labels, dim=1).numpy()
    real_labels = torch.argmax(fixed_real_labels, dim=1).cpu().numpy()

    # Get unique labels in this set
    unique_labels = np.unique(fake_labels)

    for label in unique_labels:
        # Find indices for the current label
        fake_indices = np.where(fake_labels == label)[0]
        real_indices = np.where(real_labels == label)[0]

        # Select up to 'images_per_row' images
        selected_fake = fake_indices[:images_per_row]
        selected_real = real_indices[:images_per_row]

        # If less than 'images_per_row' images are available, pad with available ones
        num_fake = len(selected_fake)
        num_real = len(selected_real)

        if num_fake < images_per_row:
            print(f"Label {label}: Only {num_fake} fake images available for visualization.")
        if num_real < images_per_row:
            print(f"Label {label}: Only {num_real} real images available for visualization.")

        # Extract selected images
        fake_imgs = fake_images[selected_fake]
        real_imgs = fixed_real_images[selected_real]

        # Create a figure with two rows: real and fake
        fig, axes = plt.subplots(2, images_per_row, figsize=(images_per_row * 2, 4))
        fig.suptitle(f'Label {label} - Set {index}', fontsize=16)

        for i in range(images_per_row):
            # Plot real image
            if i < num_real:
                ax_real = axes[0, i]
                img_real = real_imgs[i].squeeze().cpu().numpy()
                ax_real.imshow(img_real, cmap='gray')
                ax_real.axis('off')
                if i == 0:
                    ax_real.set_ylabel('Real', fontsize=12)
            else:
                axes[0, i].axis('off')

            # Plot fake image
            if i < num_fake:
                ax_fake = axes[1, i]
                img_fake = fake_imgs[i].squeeze().numpy()
                ax_fake.imshow(img_fake, cmap='gray')
                ax_fake.axis('off')
                if i == 0:
                    ax_fake.set_ylabel('Fake', fontsize=12)
            else:
                axes[1, i].axis('off')

        plt.tight_layout()
        plt.show()


# In[14]:


visualize_by_index(fake_images_list, fake_labels_list, fixed_real_images, fixed_real_labels, index=11, images_per_row=15)


# In[17]:


# In[15]:
#####################################
# Inference
#####################################

# Define the path to the saved Generator model
# Replace 'generator_epoch_X_iter_Y.pth' with the actual model file you want to load
generator_model_path = 'saved_models_0.4/generator_epoch_1751_iter_0.pth'  # Example path

# Define the directory to save inference images
inference_image_dir = 'inference_images'
os.makedirs(inference_image_dir, exist_ok=True)

# Initialize the Generator and load the saved weights
netG_inference = Generator().to(device)
netG_inference.load_state_dict(torch.load(generator_model_path, map_location=device))
netG_inference.eval()
logging.info(f"Loaded Generator model from {generator_model_path}")

# Function to perform inference and save generated images
def perform_inference(netG, dataloader, device, num_samples=5, images_per_sample=5):
    """
    Generates and saves images using the trained Generator by varying the noise.
    
    Parameters:
    - netG (nn.Module): Trained Generator model.
    - dataloader (DataLoader): DataLoader to fetch CLIP features and labels.
    - device (torch.device): Device to perform computation on.
    - num_samples (int): Number of different samples to generate.
    - images_per_sample (int): Number of images to generate per sample by varying noise.
    """
    with torch.no_grad():
        # Fetch a single batch of data
        images, clip_features, labels = next(iter(dataloader))
        images = images.to(device)
        clip_features = clip_features.to(device)
        labels = labels.to(device)
        
        # Select the first `num_samples` samples from the batch
        for sample_idx in range(num_samples):
            sample_clip = clip_features[sample_idx].unsqueeze(0).repeat(images_per_sample, 1, 1)  # [images_per_sample, 2, 512]
            sample_label = labels[sample_idx].unsqueeze(0).repeat(images_per_sample, 1)         # [images_per_sample, 2]
            
            # Generate varying noise vectors
            varied_noise = torch.randn(images_per_sample, Z_DIM, device=device)                # [images_per_sample, Z_DIM]
            
            # Generate fake images
            generated_images = netG(varied_noise, sample_clip, sample_label)                    # [images_per_sample,1,128,128]
            
            # Create a grid of generated images
            img_grid = vutils.make_grid(generated_images, nrow=images_per_sample, padding=2, normalize=True)
            
            # save the actual images
            img_real = images[sample_idx].unsqueeze(0).repeat(images_per_sample, 1, 1, 1)
            img_real_grid = vutils.make_grid(img_real, nrow=images_per_sample, padding=2, normalize=True)
            vutils.save_image(img_real_grid, os.path.join(inference_image_dir, f'real_sample_{sample_idx+1}.png'), normalize=True)
            
            # Save the image grid to the inference directory
            img_save_path = os.path.join(inference_image_dir, f'sample_{sample_idx+1}.png')
            vutils.save_image(generated_images, img_save_path, normalize=True)
            logging.info(f"Saved inference images to {img_save_path}")

# Perform inference and generate 5 samples with 5 images each
perform_inference(netG_inference, dataloader, device, num_samples=5, images_per_sample=5)


# In[ ]:





# In[ ]:




