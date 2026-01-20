# Student Assessment: Denoising Probabilistic Diffusion Models

## 📋 Overview
This assessment will test your ability to extend the concepts learned in the course. You will be tasked with generating images using a pre-trained diffusion model, evaluating the results using CLIP Score and Frechet Inception Distance, and then using MLOps tools to track and visualize your work. A new component of this assessment is to analyze the internal representations of the U-Net model to gain deeper insights into the generation process.

---

## 🎯 Goals

- **Generate high-quality images** of flowers using a text-to-image diffusion model
- **Extract intermediate embeddings** from the U-Net's downsampling path during generation
- **Evaluate generated images** using CLIP Score and FID
- **Create a FiftyOne dataset** to visualize and analyze generated images, metadata, and embeddings
- **Use FiftyOne Brain** to compute uniqueness and representativeness scores
- **Log your experiment** (hyperparameters, metrics, analysis) using Weights & Biases (wandb)

---

## ⚙️ Setup & Installation

Install the required libraries:

```bash
%pip install fiftyone wandb open-clip-torch
```

---

## 1️⃣ Image Generation & Embedding Extraction

Load the pre-trained U-Net model from notebook `05_CLIP.ipynb`, generate flower images, and extract bottleneck embeddings.

```python
import torch
from utils import UNet_utils, ddpm_utils

# Initialize the U-Net model and load pre-trained weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet_utils.UNet(
    T=400, img_ch=3, img_size=32, down_chs=(256, 256, 512), t_embed_dim=8, c_embed_dim=512
).to(device)
# model.load_state_dict(torch.load('path_to_your_model.pth'))
model.eval()

# Define text prompts
text_prompts = [
    "A photo of a red rose",
    "A photo of a white daisy",
    "A photo of a yellow sunflower"
]

# Embedding extraction using hooks
embeddings_storage = {}
def get_embedding_hook(name):
    def hook(model, input, output):
        embeddings_storage[name] = output.detach()
    return hook
# Register a forward hook on the `down2` layer
# model.down2.register_forward_hook(get_embedding_hook('down2'))

# Modify the `sample_flowers` function to generate images and store embeddings
# generated_images, _ = sample_flowers(text_prompts)
# extracted_embeddings = embeddings_storage['down2']
```

---

## 2️⃣ Evaluation: CLIP Score & FID

Evaluate the generated images using CLIP Score and FID as described below.

```python
import open_clip

# Calculate CLIP score for each image and prompt
# Use the `calculate_clip_score` function below

# Calculate FID score for the set of generated images
# Use the `calculate_fid` function and Inception model below
# Load the real TF-Flowers dataset for comparison
```

---

## 3️⃣ Embedding Analysis with FiftyOne Brain

Use FiftyOne to analyze the extracted U-Net embeddings.

```python
import fiftyone as fo
import fiftyone.brain as fob

# Create a new FiftyOne dataset
dataset = fo.Dataset(name="generated_flowers_with_embeddings")

# Add generated images to the dataset with metadata:
# - File path
# - Text prompt (as `fo.Classification` label)
# - CLIP score (custom field)
# - U-Net embedding (custom field)

# Compute uniqueness
# fob.compute_uniqueness(dataset)

# Compute representativeness using U-Net embeddings
# fob.compute_representativeness(dataset, embeddings="unet_embedding")

# Launch FiftyOne App to visualize
# session = fo.launch_app(dataset)
```

---

## 4️⃣ Logging with Weights & Biases

Log your experiment and results to Weights & Biases for tracking and comparison.

```python
import wandb

# Login to wandb
# wandb.login()

# Initialize a new wandb run
# run = wandb.init(project="diffusion_model_assessment_v2")

# Log hyperparameters (e.g., guidance weight `w`, number of steps `T`)
# Log evaluation metrics (CLIP Score and FID)

# Create a wandb.Table to log results:
# - Generated image
# - Text prompt
# - CLIP score
# - Uniqueness score
# - Representativeness score

# Finish the wandb run
# run.finish()
```

---

## 🏆 Scoring Rubric

- **20 points:** Generate images from text prompts using the model
- **15 points:** Extract intermediate embeddings from U-Net's downsampling path
- **20 points:** Calculate and report CLIP Score and FID
- **25 points:** Create a FiftyOne dataset, add embeddings, compute uniqueness and representativeness
- **20 points:** Log experiment, metrics, and analysis to Weights & Biases and publish FiftyOne dataset on HuggingFace

---

## 📏 Metric Calculation Guide

### CLIP Score
Measures semantic alignment between a text prompt and a generated image. Higher scores = better alignment.

```python
import torch
import open_clip
from PIL import Image

def calculate_clip_score(image_path, text_prompt):
    # Load model
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

    # Preprocess inputs
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    text = tokenizer([text_prompt])

    # Compute features and similarity
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Calculate dot product
        score = (image_features @ text_features.T).item()
    return score
```

> **Note:** CLIP Score must be computed on every sample and saved in the FiftyOne dataset.

### Fréchet Inception Distance (FID)
Measures distance between feature distributions of real and generated images. Lower scores = better quality/diversity.

```python
import numpy as np
from scipy.linalg import sqrtm

def calculate_fid(real_embeddings, gen_embeddings):
    # real_embeddings and gen_embeddings should be Numpy arrays of shape (N, 2048) 
    # extracted from an InceptionV3 model

    # Calculate mean and covariance
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = gen_embeddings.mean(axis=0), np.cov(gen_embeddings, rowvar=False)

    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2)

    # Calculate sqrt of product of covariances
    covmean = sqrtm(sigma1.dot(sigma2))

    # Handle numerical errors
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Final FID calculation
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    return fid
```

> **Note:** FID is computed once for the entire set of generated images against real images.

---

## 🎁 Bonus Task: Classifier with "IDK" Option (20 points)

Modify an MNIST-digit classifier to include an "I don't know" (IDK) option. This allows the model to defer to a human when not confident.

- Update the MNIST generation and classification notebook
- Create and publish a FiftyOne dataset with your experiment results
