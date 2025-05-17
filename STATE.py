#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import argparse
import logging
from pathlib import Path

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
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

#####################################
# Model Definitions
#####################################

class CombinedBinaryClassifier(nn.Module):
    """
    Threat alignment classifier that processes:
    - A 128×128 binary trajectory image 
    - Five CLIP vectors (each 512-dim)
    
    Architecture:
    - Flattens the trajectory image and maps it to hidden_dim
    - Processes each of the 5 CLIP vectors to hidden_dim
    - Concatenates all six hidden_dim vectors, fuses to fusion_dim, then outputs logits
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
        Forward pass of the classifier
        
        Args:
            clip_feats: Tensor of shape [B, C, 512] where C is number of CLIP vectors (typically 2)
            traj_img: Tensor of shape [B, 1, 128, 128] representing trajectory images
            
        Returns:
            Tensor of shape [B] containing logits for binary classification
        """
        B, C, D = clip_feats.shape
        num_procs = len(self.feature_processors)  # ==5

        # If we got fewer than 5 CLIP vectors, pad with zeros
        if C < num_procs:
            pad = torch.zeros(B, num_procs - C, D,
                              device=clip_feats.device,
                              dtype=clip_feats.dtype)
            clip_feats = torch.cat([clip_feats, pad], dim=1)

        out_feats = []
        for i, proc in enumerate(self.feature_processors):
            out_feats.append(proc(clip_feats[:, i, :]))  # Process each CLIP vector

        img_feat = self.img_processor(traj_img)           # [B, hidden_dim]
        out_feats.append(img_feat)

        x = torch.cat(out_feats, dim=1)
        x = self.fusion(x)
        logits = self.classifier(x)
        return logits.view(-1)

class Generator(nn.Module):
    """
    Generator for the STATE model.
    
    Takes noise vectors, CLIP features, and class labels to generate trajectory images.
    The architecture processes each input separately, then combines them before 
    generating the final image through transposed convolutions.
    """
    def __init__(self, z_dim, clip_num_vectors, clip_vector_dim, label_dim, hidden_dim, image_channel):
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
        # Total features after concatenation: noise + clip_num_vectors CLIP vectors + labels
        self.combine_fc = nn.Sequential(
            nn.Linear(hidden_dim * 8 * 4 * 4 * (1 + clip_num_vectors + 1), hidden_dim * 8 * 4 * 4),
            nn.BatchNorm1d(hidden_dim * 8 * 4 * 4),
            nn.ReLU(True)
        )

        # Main generator network
        self.main = nn.Sequential(
            # Initial size: [B, hidden_dim*8, 4, 4]
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
        """
        Forward pass of the generator
        
        Args:
            noise: Tensor of shape [B, z_dim]
            clip_features: Tensor of shape [B, clip_num_vectors, clip_vector_dim]
            labels: Tensor of shape [B, label_dim]
            
        Returns:
            Tensor of shape [B, image_channel, 128, 128] containing generated images
        """
        B = noise.size(0)

        # Process noise
        noise_processed = self.noise_fc(noise)  # [B, hidden_dim*8*4*4]

        # Process each CLIP vector independently
        clip_processed = []
        for i in range(self.clip_num_vectors):
            clip_vector = clip_features[:, i, :]  # [B, clip_vector_dim]
            processed = self.clip_fcs[i](clip_vector)  # [B, hidden_dim*8*4*4]
            clip_processed.append(processed)
        clip_processed = torch.cat(clip_processed, dim=1)  # [B, hidden_dim*8*4*4 * clip_num_vectors]

        # Process labels
        label_processed = self.label_fc(labels)  # [B, hidden_dim*8*4*4]

        # Combine all processed features
        combined = torch.cat([noise_processed, clip_processed, label_processed], dim=1)  
        # [B, hidden_dim*8*4*4 * (1 + clip_num_vectors + 1)]

        # Further process combined features
        combined = self.combine_fc(combined)  # [B, hidden_dim*8*4*4]

        # Reshape to [B, hidden_dim*8, 4, 4]
        combined = combined.view(-1, self.hidden_dim * 8, 4, 4)

        # Pass through main generator layers
        output = self.main(combined)  # [B, image_channel, 128, 128]

        return output


class Discriminator(nn.Module):
    """
    Discriminator for the STATE model.
    
    Takes trajectory images, CLIP features, and class labels to discriminate
    between real and fake trajectories. The architecture processes CLIP and label
    data, expands them to match image dimensions, then concatenates with the image
    for convolutional processing.
    """
    def __init__(self, clip_num_vectors, clip_vector_dim, label_dim, hidden_dim, image_channel, image_size):
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
            nn.Linear(self.clip_dim, hidden_dim * 2),  # Compress from clip_dim to hidden_dim*2
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim * 2, hidden_dim),  # Compress to hidden_dim
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
        # Input channels: image_channel + hidden_dim (compressed CLIP) + hidden_dim (labels)
        self.main = nn.Sequential(
            nn.Conv2d(image_channel + hidden_dim + hidden_dim, hidden_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Final layer to reduce to [B,1,1,1]
            nn.Conv2d(hidden_dim * 8, 1, 8, 8, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, img, clip_features, labels):
        """
        Forward pass of the discriminator
        
        Args:
            img: Tensor of shape [B, image_channel, image_size, image_size]
            clip_features: Tensor of shape [B, clip_num_vectors, clip_vector_dim]
            labels: Tensor of shape [B, label_dim]
            
        Returns:
            Tensor of shape [B] containing discrimination scores
        """
        B = img.size(0)

        # Flatten CLIP features
        clip_features_flat = clip_features.view(B, -1)  # [B, clip_dim]

        # Compress CLIP features
        clip_compressed = self.clip_compression(clip_features_flat)  # [B, hidden_dim]

        # Process labels
        label_processed = self.label_proj(labels)  # [B, hidden_dim]

        # Expand compressed CLIP features and labels to match image spatial dimensions
        clip_expanded = clip_compressed.unsqueeze(-1).unsqueeze(-1)  # [B, hidden_dim, 1, 1]
        clip_expanded = clip_expanded.repeat(1, 1, self.image_size, self.image_size)  
        # [B, hidden_dim, image_size, image_size]

        label_expanded = label_processed.unsqueeze(-1).unsqueeze(-1)      # [B, hidden_dim, 1, 1]
        label_expanded = label_expanded.repeat(1, 1, self.image_size, self.image_size)  
        # [B, hidden_dim, image_size, image_size]

        # Concatenate image, compressed CLIP features, and labels
        x = torch.cat([img, clip_expanded, label_expanded], dim=1)   
        # [B, image_channel + hidden_dim + hidden_dim, image_size, image_size]

        # Pass through convolutional layers
        output = self.main(x)  # [B,1,1,1]
        output = output.view(-1)  # [B]

        return output

#####################################
# Dataset Definition
#####################################
class TrajectoryDataset(Dataset):
    """
    Dataset for loading trajectory images, CLIP features, and threat labels.
    
    Each trajectory consists of:
    - A binary trajectory image
    - CLIP feature vectors for the trajectory
    - A binary threat label (threatening/non-threatening)
    """
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
            logger.info(f"Loaded {len(annotations)} annotations from {json_file}.")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}")
            raise
        except FileNotFoundError:
            logger.error(f"JSON file not found: {json_file}")
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

        logger.info(f"Extracted max threat scores for {len(threat_scores)} unique trajectory IDs.")

        # Binarize the scores based on the threshold
        binarized_scores = {traj_id: (1 if score >= self.threshold else 0) 
                            for traj_id, score in threat_scores.items()}
        logger.info(f"Binarized threat scores using threshold {self.threshold}.")

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
                        logger.warning(f"No binarized score for trajectory ID {traj_id}. Skipping.")
                else:
                    logger.warning(f"Missing files in {traj_dir}. Expected .png and _CLIP.npy files.")
        logger.info(f"Collected {len(samples)} valid samples.")
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
                   - clip_features (torch.Tensor): Tensor of shape [num_vectors, vector_dim].
                   - one_hot_label (torch.Tensor): One-hot encoded label (2 classes).
        """
        trajectory_id, traj_dir = self.samples[index]

        # Load trajectory image
        traj_img_path = os.path.join(traj_dir, f"{trajectory_id}_trajectory.png")
        try:
            img = Image.open(traj_img_path).convert('L')  # Convert to grayscale
        except Exception as e:
            logger.error(f"Error loading image {traj_img_path}: {e}")
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
            logger.error(f"Error loading CLIP numpy file {traj_clip_path}: {e}")
            raise

        # Convert to torch tensor
        clip_features_tensor = torch.tensor(clip_features, dtype=torch.float32)  # Shape: [2, 512]

        # Get the binary threat label and convert to one-hot
        label_binary = self.binarized_scores[trajectory_id]
        one_hot_label = torch.zeros(2, dtype=torch.float32)
        one_hot_label[label_binary] = 1.0

        return img, clip_features_tensor, one_hot_label

#####################################
# Utility Functions
#####################################
def weights_init(m):
    """Initialize weights for Conv and BatchNorm layers."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def get_fixed_batch(dataloader, device, batch_size):
    """Get a fixed batch for visualization from the dataloader."""
    for data in dataloader:
        images, clip_features, labels = data
        images = images.to(device)
        clip_features = clip_features.to(device)
        labels = labels.to(device)
        return images[:batch_size], clip_features[:batch_size], labels[:batch_size]

def visualize_batch(images, nrow=8, title=None, save_path=None):
    """Visualize a batch of images."""
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    if title:
        plt.title(title)
    plt.imshow(np.transpose(vutils.make_grid(images.cpu(), nrow=nrow, padding=2, normalize=True), (1, 2, 0)))
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def visualize_samples_by_label(fake_images, labels, images_per_row=8, save_path=None):
    """
    Visualize generated samples grouped by their label.
    
    Args:
        fake_images: Tensor of shape [B, C, H, W] with generated images
        labels: Tensor of shape [B, label_dim] with one-hot labels
        images_per_row: Number of images to display per row
        save_path: Optional path to save the visualization
    """
    # Convert labels to integer indices
    label_indices = torch.argmax(labels, dim=1).cpu().numpy()
    
    # Get unique labels
    unique_labels = np.unique(label_indices)
    
    fig, axes = plt.subplots(len(unique_labels), 1, figsize=(15, 5*len(unique_labels)))
    if len(unique_labels) == 1:
        axes = [axes]
    
    for i, label in enumerate(unique_labels):
        # Get images for this label
        mask = (label_indices == label)
        label_images = fake_images[mask]
        
        if len(label_images) == 0:
            continue
            
        # Only take up to images_per_row images
        label_images = label_images[:images_per_row]
        
        # Create grid for this label
        grid = vutils.make_grid(label_images, nrow=images_per_row, padding=2, normalize=True)
        
        # Display in the appropriate subplot
        axes[i].imshow(np.transpose(grid, (1, 2, 0)))
        axes[i].set_title(f"Label {label}")
        axes[i].axis("off")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def perform_inference(netG, dataloader, device, output_dir, num_samples=5, images_per_sample=5):
    """
    Generate and save trajectory images using the trained Generator.
    
    Args:
        netG: Trained Generator model
        dataloader: DataLoader to fetch CLIP features and labels
        device: Device to perform computation on
        output_dir: Directory to save generated images
        num_samples: Number of different samples to generate
        images_per_sample: Number of images to generate per sample by varying noise
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        # Fetch a single batch of data
        images, clip_features, labels = next(iter(dataloader))
        images = images.to(device)
        clip_features = clip_features.to(device)
        labels = labels.to(device)
        
        # Select the first `num_samples` samples from the batch
        for sample_idx in range(min(num_samples, len(images))):
            # Create a directory for this sample
            sample_dir = os.path.join(output_dir, f"sample_{sample_idx+1}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # Get the real image and save it
            real_img = images[sample_idx:sample_idx+1]
            real_path = os.path.join(sample_dir, "real.png")
            vutils.save_image(real_img, real_path, normalize=True)
            
            # Repeat the CLIP and label tensors
            sample_clip = clip_features[sample_idx].unsqueeze(0).repeat(images_per_sample, 1, 1)
            sample_label = labels[sample_idx].unsqueeze(0).repeat(images_per_sample, 1)
            
            # Create a label description for the filename
            label_desc = "threatening" if torch.argmax(sample_label[0]).item() == 1 else "non_threatening"
            
            # Generate varying noise vectors
            varied_noise = torch.randn(images_per_sample, netG.z_dim, device=device)
            
            # Generate fake images
            generated_images = netG(varied_noise, sample_clip, sample_label)
            
            # Save individual generated images
            for img_idx in range(images_per_sample):
                img_path = os.path.join(sample_dir, f"generated_{img_idx+1}.png")
                vutils.save_image(generated_images[img_idx:img_idx+1], img_path, normalize=True)
            
            # Save a grid of all generated images for this sample
            grid_path = os.path.join(sample_dir, f"grid_{label_desc}.png")
            vutils.save_image(generated_images, grid_path, nrow=images_per_sample, normalize=True)
            
            logger.info(f"Saved inference images for sample {sample_idx+1} to {sample_dir}")

#####################################
# Training Function
#####################################
def train(args):
    """
    Train the STATE model using the specified arguments.
    
    Args:
        args: Namespace containing training arguments
    """
    # Set up device
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    
    logger.info(f"Using device: {device}")
    if use_cuda:
        logger.info(f"CUDA version: {torch.version.cuda}")
    
    # Create output directories
    model_save_dir = os.path.join(args.output_dir, "models")
    image_save_dir = os.path.join(args.output_dir, "images")
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(image_save_dir, exist_ok=True)
    
    # Set up transforms
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Create dataset and dataloader
    dataset = TrajectoryDataset(
        root=args.data_dir,
        json_file=args.annotations_file,
        threshold=args.threshold,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    
    logger.info(f"Created dataloader with {len(dataset)} samples")
    
    # Load threat-alignment classifier if specified
    if args.threat_model_path:
        netT = CombinedBinaryClassifier().to(device)
        netT.load_state_dict(torch.load(args.threat_model_path, map_location=device))
        netT.eval()
        for p in netT.parameters():
            p.requires_grad = False
        logger.info(f"Loaded threat classifier from {args.threat_model_path}")
        criterion_cls = nn.BCELoss()
    else:
        netT = None
    
    # Initialize generator and discriminator
    netG = Generator(
        z_dim=args.z_dim,
        clip_num_vectors=args.clip_num_vectors,
        clip_vector_dim=args.clip_vector_dim,
        label_dim=args.label_dim,
        hidden_dim=args.g_hidden,
        image_channel=args.image_channel
    ).to(device)
    
    netD = Discriminator(
        clip_num_vectors=args.clip_num_vectors,
        clip_vector_dim=args.clip_vector_dim,
        label_dim=args.label_dim,
        hidden_dim=args.d_hidden,
        image_channel=args.image_channel,
        image_size=args.image_size
    ).to(device)
    
    # Initialize weights
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    logger.info("Initialized generator and discriminator")
    
    # Set up loss function and optimizers
    criterion = nn.BCELoss()
    
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # Get a fixed batch for visualization
    fixed_real_images, fixed_real_clip, fixed_real_labels = get_fixed_batch(
        dataloader, device, args.batch_size
    )
    
    # Sample fixed noise
    fixed_noise = torch.randn(args.batch_size, args.z_dim, device=device)
    
    # Initialize lists to store losses and samples
    G_losses = []
    D_losses = []
    fake_images_list = []
    fake_labels_list = []
    iters = 0
    
    logger.info("Starting training...")
    
    for epoch in range(args.epochs):
        for i, (images, clip_features, labels) in enumerate(dataloader):
            images = images.to(device)
            clip_features = clip_features.to(device)
            labels = labels.to(device)
            b_size = images.size(0)
            
            ############################################################
            # (1) Update D: maximize log(D(x)) + log(1 - D(G(z,h,l)))
            ############################################################
            netD.zero_grad()
            
            # Train with real data
            real_label_tensor = torch.full((b_size,), args.real_label, dtype=torch.float, device=device)
            output_real = netD(images, clip_features, labels)
            errD_real = criterion(output_real, real_label_tensor)
            errD_real.backward()
            D_x = output_real.mean().item()
            
            # Train with fake data
            noise = torch.randn(b_size, args.z_dim, device=device)
            fake_images = netG(noise, clip_features, labels)
            fake_label_tensor = torch.full((b_size,), args.fake_label, dtype=torch.float, device=device)
            output_fake = netD(fake_images.detach(), clip_features, labels)
            errD_fake = criterion(output_fake, fake_label_tensor)
            errD_fake.backward()
            D_G_z1 = output_fake.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()
            
            ############################################################
            # (2) Update G: maximize log(D(G(z,h,l)))
            ############################################################
            netG.zero_grad()
            
            # Adversarial loss
            output_fake_for_G = netD(fake_images, clip_features, labels)
            adv_loss = criterion(output_fake_for_G, real_label_tensor)
            D_G_z2 = output_fake_for_G.mean().item()
            
            if netT is not None:
                # Threat-alignment loss
                cls_logits = netT(clip_features, fake_images)
                cls_prob = torch.sigmoid(cls_logits).view(-1)
                target_scalar = labels[:, 1]  # 1 for "threatening", 0 otherwise
                cls_loss = criterion_cls(cls_prob, target_scalar)
                
                # Combined loss
                errG = args.lambda_v * adv_loss + args.lambda_t * cls_loss
            else:
                # Only adversarial loss if no threat model
                errG = adv_loss
            
            errG.backward()
            optimizerG.step()
            
            # Save losses
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            # Output training stats
            if i % args.log_interval == 0:
                logger.info(
                    f'[{epoch+1}/{args.epochs}][{i}/{len(dataloader)}] '
                    f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                    f'D(x): {D_x:.4f} D(G(z,h,l)): {D_G_z1:.4f}/{D_G_z2:.4f}'
                )
            
            # Check generator progress
            if (iters % args.sample_interval == 0) or ((epoch == args.epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    # Generate images using fixed inputs
                    fake_images_fixed = netG(fixed_noise, fixed_real_clip, fixed_real_labels).detach().cpu()
                
                # Store images and labels for visualization
                fake_images_list.append(fake_images_fixed)
                fake_labels_list.append(fixed_real_labels.cpu())
                
                # Save grid of generated images
                img_path = os.path.join(image_save_dir, f'epoch_{epoch+1}_iter_{i}.png')
                vutils.save_image(fake_images_fixed, img_path, normalize=True)
                logger.info(f"Saved generated images to {img_path}")
                
                # Visualize by label if enabled
                if args.visualize_by_label:
                    label_img_path = os.path.join(image_save_dir, f'epoch_{epoch+1}_iter_{i}_by_label.png')
                    visualize_samples_by_label(fake_images_fixed, fixed_real_labels.cpu(), save_path=label_img_path)
            
            iters += 1
        
        # Save model every N epochs
        if (epoch + 1) % args.save_interval == 0:
            model_path = os.path.join(model_save_dir, f'generator_epoch_{epoch+1}.pth')
            torch.save(netG.state_dict(), model_path)
            logger.info(f"Saved generator to {model_path}")
    
    # Save final model
    final_model_path = os.path.join(model_save_dir, 'generator_final.pth')
    torch.save(netG.state_dict(), final_model_path)
    logger.info(f"Saved final generator to {final_model_path}")
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(image_save_dir, 'loss_plot.png'))
    logger.info(f"Saved loss plot to {os.path.join(image_save_dir, 'loss_plot.png')}")
    
    return {
        'netG': netG,
        'netD': netD,
        'G_losses': G_losses,
        'D_losses': D_losses,
        'fake_images_list': fake_images_list,
        'fake_labels_list': fake_labels_list
    }

#####################################
# Inference Function
#####################################
def run_inference(args):
    """
    Run inference with a trained generator model.
    
    Args:
        args: Namespace containing inference arguments
    """
    # Set up device
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    logger.info(f"Using device: {device}")
    
    # Set up transforms
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Create dataset and dataloader for sampling CLIP features and labels
    dataset = TrajectoryDataset(
        root=args.data_dir,
        json_file=args.annotations_file,
        threshold=args.threshold,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    
    logger.info(f"Created dataloader with {len(dataset)} samples")
    
    # Initialize generator
    netG = Generator(
        z_dim=args.z_dim,
        clip_num_vectors=args.clip_num_vectors,
        clip_vector_dim=args.clip_vector_dim,
        label_dim=args.label_dim,
        hidden_dim=args.g_hidden,
        image_channel=args.image_channel
    ).to(device)
    
    # Load trained weights
    netG.load_state_dict(torch.load(args.generator_path, map_location=device))
    netG.eval()
    logger.info(f"Loaded generator from {args.generator_path}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Perform inference
    perform_inference(
        netG=netG,
        dataloader=dataloader,
        device=device,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        images_per_sample=args.images_per_sample
    )
    
    logger.info(f"Inference completed. Results saved to {args.output_dir}")

#####################################
# Main Function
#####################################
def main():
    parser = argparse.ArgumentParser(description='STATE: Trajectory Generation with GAN')
    
    subparsers = parser.add_subparsers(dest='mode', help='Mode: train or inference')
    
    # Training arguments
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory for trajectory data')
    train_parser.add_argument('--annotations_file', type=str, required=True,
                        help='Path to JSON file with threat annotations')
    train_parser.add_argument('--output_dir', type=str, default='state_output',
                        help='Directory to save models and images')
    train_parser.add_argument('--threshold', type=int, default=3,
                        help='Threshold for binarizing threat scores. DEWS paper used: LTP=3, MTP=6, HTP=8')
    train_parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    train_parser.add_argument('--image_channel', type=int, default=1,
                        help='Number of image channels (1 for grayscale)')
    train_parser.add_argument('--image_size', type=int, default=128,
                        help='Size of the images (both height and width)')
    train_parser.add_argument('--z_dim', type=int, default=100,
                        help='Dimension of random noise input')
    train_parser.add_argument('--clip_num_vectors', type=int, default=2,
                        help='Number of CLIP vectors to use per sample')
    train_parser.add_argument('--clip_vector_dim', type=int, default=512,
                        help='Dimension of each CLIP vector')
    train_parser.add_argument('--label_dim', type=int, default=2,
                        help='Dimension of labels (2 for binary classification)')
    train_parser.add_argument('--g_hidden', type=int, default=64,
                        help='Hidden dimension multiplier for generator')
    train_parser.add_argument('--d_hidden', type=int, default=64,
                        help='Hidden dimension multiplier for discriminator')
    train_parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of training epochs')
    train_parser.add_argument('--real_label', type=float, default=1.0,
                        help='Label value for real images')
    train_parser.add_argument('--fake_label', type=float, default=0.0,
                        help='Label value for fake images')
    train_parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    train_parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    train_parser.add_argument('--save_interval', type=int, default=50,
                        help='Interval for saving model checkpoints (epochs)')
    train_parser.add_argument('--sample_interval', type=int, default=500,
                        help='Interval for sampling and saving images (iterations)')
    train_parser.add_argument('--log_interval', type=int, default=50,
                        help='Interval for logging training progress (iterations)')
    train_parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    train_parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA if available')
    train_parser.add_argument('--visualize_by_label', action='store_true',
                        help='Visualize samples grouped by label')
    train_parser.add_argument('--threat_model_path', type=str, default=None,
                        help='Path to pretrained threat classifier model')
    train_parser.add_argument('--lambda_v', type=float, default=0.6,
                        help='Weight for adversarial validity loss')
    train_parser.add_argument('--lambda_t', type=float, default=0.4,
                        help='Weight for threat-alignment loss')
    
    # Inference arguments
    infer_parser = subparsers.add_parser('inference', help='Run inference with a trained model')
    infer_parser.add_argument('--generator_path', type=str, required=True,
                        help='Path to trained generator model')
    infer_parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory for trajectory data (for sampling CLIP and labels)')
    infer_parser.add_argument('--annotations_file', type=str, required=True,
                        help='Path to JSON file with threat annotations')
    infer_parser.add_argument('--output_dir', type=str, default='inference_output',
                        help='Directory to save generated images')
    infer_parser.add_argument('--threshold', type=int, default=3,
                        help='Threshold for binarizing threat scores')
    infer_parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for data loading')
    infer_parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of different samples to generate')
    infer_parser.add_argument('--images_per_sample', type=int, default=5,
                        help='Number of variations to generate per sample')
    infer_parser.add_argument('--image_channel', type=int, default=1,
                        help='Number of image channels (1 for grayscale)')
    infer_parser.add_argument('--image_size', type=int, default=128,
                        help='Size of the images (both height and width)')
    infer_parser.add_argument('--z_dim', type=int, default=100,
                        help='Dimension of random noise input')
    infer_parser.add_argument('--clip_num_vectors', type=int, default=2,
                        help='Number of CLIP vectors to use per sample')
    infer_parser.add_argument('--clip_vector_dim', type=int, default=512,
                        help='Dimension of each CLIP vector')
    infer_parser.add_argument('--label_dim', type=int, default=2,
                        help='Dimension of labels (2 for binary classification)')
    infer_parser.add_argument('--g_hidden', type=int, default=64,
                        help='Hidden dimension multiplier for generator')
    infer_parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    infer_parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA if available')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        run_inference(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()




