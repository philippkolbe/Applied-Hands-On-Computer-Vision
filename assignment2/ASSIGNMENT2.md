## Links
- [wandb project](https://wandb.ai/hpi-philipp-kolbe/cilp-extended-assessment)
- [Dataset](https://huggingface.co/datasets/philippkolbe/cilp_assessment_subset)


## Dataset
As we will do a large number of experiments for hyperparameter search, avoid training on the whole dataset. We select a subset of image-lidar pairs (e.g. 500), created a FiftyOne dataset on HuggingFace, and used it for all experiments.


## Submission Checklist
Before submitting, verify:

- All notebooks run in Google Colab without errors
- W&B project is public and link is in README
- FiftyOne screenshots are in results/ folder
- All fusion architectures are implemented and compared
- EmbedderStrided class is complete and tested
- README.md contains all required sections
- requirements.txt lists all dependencies
- Model checkpoints are saved in checkpoints/



## Task 1: Experiment Tracking with Weights & Biases (20 points)

### Objective
Integrate W&B into your training pipeline to track experiments, compare runs, and visualize results.

#### 1.1 Project Setup (4 points)
Create a W&B project named cilp-extended-assessment
Log all training runs to this project
Include your W&B username in your submission

#### 1.2 Metric Logging (8 points)

| Criterion | Points | Requirements |
|---|---:|---|
| Project setup | 4 | W&B project created and accessible |
| Metric logging | 8 | All required metrics logged correctly |
| Visualization logging | 2 | Loss curves in W&B |
| Hyperparameter config | 4 | All hyperparameters logged as config |
Number of parameters
Once per run (config)

#### 1.3 Visualization Logging (4 points)
Log these visualizations to W&B:

- Training/validation loss curves
- Sample predictions (at least 5 examples)
- Similarity matrix at the end of training (for contrastive learning tasks only - see Task 5)

#### 1.4 Hyperparameter Configuration (4 points)
Log hyperparameters as W&B config:

- Embedding size in CILP_Model, MyMultimodalModel (rgb_net, xyz_net don’t need this, but you can log the final layer size)
- Learning rate
- Batch size
- Number of epochs
- Optimizer type (e.g., SGD with momentum, Adam, AdamW)
- Fusion strategy (for Task 3, use “N/A” or “single_modality” for other tasks)

#### Deliverables
Public W&B project link in your README
Screenshots of your W&B dashboard showing logged metrics

Recommendation on what to log for each task:

| Metric | Task 2 | Task 3 | Task 4 | Task 5 |
|---|:---:|:---:|:---:|:---:|
| Training loss | N/A | Every epoch | Every epoch | Every epoch |
| Validation loss | N/A | Every epoch | Every epoch | Every epoch |
| Learning rate | N/A | Config or per epoch / batch (if using LR scheduler) | Config or per epoch (if using LR scheduler) | Config or per epoch / batch (if using LR scheduler) |
| Batch size | N/A | Config | Config | Config |
| Embedding size | N/A | Skip or final layer size | Skip or final layer size | 200 (config) |
| Fusion strategy | N/A | "late"/"intermediate" | Same as Task 3 | "contrastive" or skip |
| Similarity matrix | N/A | Skip | Skip | End of CILP training |



## Task 2: Dataset Visualization with FiftyOne (15 points)
| Criterion | Points | Requirements |
|---|---:|---|
| Grouped dataset | 6 | Correct structure with RGB and LiDAR groups |
| Statistics | 4 | Complete dataset statistics documented |
| Exploration | 5 | Screenshots and written observations |
Create an interactive visualization of the CILP dataset showing RGB images and LiDAR data side-by-side.

### Dataset
Use the assessment dataset (data/assessment/) for Tasks 2-5 to maintain consistency. This enables accuracy metrics required in Task 4.

#### 2.1 Grouped Dataset Creation (6 points)
Create a FiftyOne dataset with:
- Group field linking RGB and LiDAR modalities
- Class labels (cube, sphere) as classification fields
- File paths for both modalities

# Expected structure

```python
dataset = fo.Dataset("cilp_assessment")

dataset.add_group_field("group", default="rgb")
```

#### 2.2 Dataset Statistics (4 points)
Include in your notebook:

- Total number of samples per class
- Train/validation split sizes
- Image dimensions and data types
- Class distribution visualization (bar chart or histogram)

#### 2.3 Visual Exploration (5 points)
Document your findings:
- Take screenshots of FiftyOne showing RGB and LiDAR pairs
- Identify at least 3 observations about the dataset characteristics
- Note any data quality issues or patterns you observe

#### Deliverables
- FiftyOne dataset creation code in your notebook
- Screenshots of the FiftyOne interface
- Written observations about dataset characteristics

## Task 3: Fusion Architecture Comparison (20 points)
### Objective
Implement and compare two fusion strategies (late and intermediate) for combining RGB and LiDAR embeddings. For intermediate fusion, you will implement three different merging methods (concatenation, addition, Hadamard product) for a total of four architectures to compare. Note that this will be a classification task (cubes vs spheres) instead of the regression task in the original NVIDIA lab notebooks.

### Architectures to Implement
### Task 3: Fusion Comparison (20 points)

| Criterion | Points | Requirements |
|---|---:|---|
| Late fusion | 7.5 | Working implementation with results |
| Intermediate fusion | 7.5 | Working implementation with results |
| Analysis | 5 | Complete comparison table and written analysis |

#### 3.1 Late Fusion (7.5 points)
Process modalities separately, combine outputs:
- Separate encoders for RGB (4 channels) and LiDAR (4 channels)
- Concatenate final embeddings before passing to classifier head
- Document advantages and limitations

Suggested Architecture:

- RGB Encoder: Conv layers → embedding (e.g., 100-dimensional)
- LiDAR Encoder: Conv layers → embedding (e.g., 100-dimensional)
- Concatenated: 200-dimensional combined embedding
- Classifier Head: Linear(200, 100) → ReLU → Linear(100, 1) → BCEWithLogitsLoss

#### 3.2 Intermediate Fusion (7.5 points)
Combine feature maps at an intermediate layer:

- Separate early convolutions for each modality
- Concatenate at the feature map level (after conv2 or conv3)
- Shared later layers process combined features
- Document advantages and limitations

Suggested Architecture:

```python
RGB Conv Path: Conv2d(4, 50) → ReLU → MaxPool2d → Conv2d(50, 100) → ReLU → MaxPool2d
LiDAR Conv Path: Conv2d(4, 50) → ReLU → MaxPool2d → Conv2d(50, 100) → ReLU → MaxPool2d
Merge: Concatenate/Add/Multiply feature maps here
Shared Layers: Conv2d → ... → Flatten → Classifier Head
```

#### 3.3 Intermediate Fusion Variants (Included in 7.5 points above)
Implement all three methods for combining feature maps:

- `torch.cat((x_rgb, x_lidar), dim=1)` (Concatenation)
    - Doubles the channel dimension
    - Preserves all information from both modalities
    - Requires adjusting next layer's input channels

- `x_rgb + x_lidar` (Addition)
    - Requires same channel dimensions for both modalities
    - Element-wise addition combines features
    - Maintains channel dimension

- `x_rgb * x_lidar` (Hadamard Product)
    - Element-wise multiplication
    - Emphasizes areas where both modalities have strong activations
    - Maintains channel dimension

#### 3.4 Comparison Analysis (5 points)

#### Deliverables
- Implementation of late fusion architecture
- Implementation of all three intermediate fusion variants (concatenation, addition, Hadamard product)
- Comparison table with measured metrics for all four architectures

| Metric | Late Fusion | Intermediate (Concat) | Intermediate (Add) | Intermediate (Hadamard) |
|---|---:|---:|---:|---:|
| Validation Loss |  |  |  |  |
| F1 score |  |  |  |  |
| Parameters (count) |  |  |  |  |
| Training Time (seconds/epoch) |  |  |  |  |
| GPU Memory (MB) |  |  |  |  |

- Written analysis of results (200-400 words)

### Task 4: Strided Convolution Ablation (20 points)

#### Objective
Replace MaxPool2d layers with stride-2 convolutions and analyze the impact. Note that we will use the same fusion models as in task 3. Create two versions of that architecture (MaxPool2d vs Strided Conv) and compare them.

#### Background
The original Embedder architecture uses MaxPool2d for spatial downsampling:

```python
nn.Conv2d(in_ch, 50, kernel_size=3, padding=1),

nn.ReLU(),

nn.MaxPool2d(2),  # Replace this
```

#### 4.1 Strided Convolution Implementation (8 points)
Replace each MaxPool2d with a stride-2 convolution:

```python
# Original
nn.Conv2d(50, 100, kernel_size=3, padding=1),

nn.ReLU(),

nn.MaxPool2d(2),

# Modified
nn.Conv2d(50, 100, kernel_size=3, stride=2, padding=1),

nn.ReLU(),
```

Implement a new EmbedderStrided class with this modification applied to all pooling layers.

#### 4.2 Performance Comparison (6 points)
Train both architectures (MaxPool and Strided) with identical hyperparameters and compare:

| Metric | MaxPool2d | Strided Conv | Difference |
|---|---:|---:|---:|
| Validation Loss |  |  |  |
| Parameters |  |  |  |
| Training Time |  |  |  |
| Final Accuracy |  |  |  |

#### 4.3 Analysis (6 points)
Write an analysis (200-300 words) covering:
- Quantitative comparison of results
- Theoretical differences between the two approaches
- Impact on gradient flow and learned features
- Recommendation with justification

##### Deliverables
- EmbedderStrided class implementation
- Side-by-side training results logged to W&B
- Comparison table and written analysis


## Task 5: CILP Assessment Performance (15 points)
### Objective
Complete the CILP model training and achieve the required performance thresholds.

### Requirements
#### 5.1 Contrastive Pretraining (7 points)
Train the CILP model to achieve:

- Validation loss below 3.5 (5 points)
- Validation loss below 3.2 (2 bonus points)

- Use the provided Embedder architecture (lines 41-67 in 05_Assessment.ipynb) OR apply insights from Task 3 regarding optimal depth and complexity

- Apply your best-performing downsampling method from Task 4 (MaxPool2d or Strided Conv)

- Log all training to W&B

#### 5.2 Cross-Modal Projector (4 points)
Train the projector to map RGB embeddings to LiDAR embedding space:
- MSE loss below 2.5 on validation set
- Document projector architecture and training details

#### 5.3 Final Classifier Accuracy (4 points)
The RGB-to-LiDAR classifier must achieve:
- Validation accuracy above 95%
- Test on at least 5 validation batches

#### Deliverables
- Trained model achieving required metrics
- W&B logs showing loss curves and final performance
- Model checkpoint files

## Task 6: Code Quality and Documentation (10 points)

### Requirements
#### 6.1 Code Organization (4 points)
- Modular code structure following the repository template
- Clear separation of models, datasets, training, and visualization
- No code duplication

#### 6.2 Documentation (3 points)
- README.md explaining how to run your code
- Docstrings for all functions and classes
- Comments explaining non-obvious logic

#### 6.3 Reproducibility (3 points)
- requirements.txt with pinned versions
- Random seed management for reproducible results
- Clear instructions for data setup*

As we are doing a lot of hyperparameter search, we won't use the whole dataset for every experiment. Document this by creating a subset of the used data on HuggingFace datasets or Google Drive.

## Submission Requirements
### Repository Structure
Submit a GitHub repository with this structure:

```
your-repo-name/

├── notebooks/

│   ├── 01_dataset_exploration.ipynb    # Task 2

│   ├── 02_fusion_comparison.ipynb      # Task 3

│   ├── 03_strided_conv_ablation.ipynb  # Task 4

│   └── 04_final_assessment.ipynb       # Task 5

│

├── src/

│   ├── __init__.py

│   ├── models.py          # All model architectures

│   ├── datasets.py        # Dataset classes

│   ├── training.py        # Training loops

│   └── visualization.py   # Plotting utilities

│

├── checkpoints/           # Saved model weights

├── results/               # Figures and tables

├── requirements.txt       # Dependencies

└── README.md              # Setup and usage instructions
```

## Notebook Requirements
All notebooks must:
- Run in Google Colab without modification
- Include markdown cells explaining each section
- Display all figures inline
- Print final metrics at the end

## README.md Contents
Your README must include:

- Project title and brief description
- Setup instructions (Colab or local)
- W&B project link
- Summary of results (table format)
- Instructions to reproduce results
- Any known issues or limitations

### Working with Structured Projects in Google Colab
Working with structured projects that are shared in Github in Colab requires additional preparation, please refer to the following guide and report early any issues that you might encounter. You are allowed to submit the assessment through any Github repository even if the model is not trained on Google Colab. Please refer to the following guide to enable GPU training in Kaggle notebooks.

### Required Artifacts
- Artifact
- Location
- Format
- W&B project link
- README.md
- URL
- FiftyOne screenshots
- results/
- PNG
- Comparison tables
- Notebooks
- Markdown
- Model checkpoints
- checkpoints/
- .pt files
- Final metrics summary
- README.md
- Markdown table



### Technical Specifications
- Environment
- Python 3.11 or higher
- PyTorch 2.0 or higher
- CUDA-capable GPU (recommended)
- Data Access

The dataset should be structured as:
```
data/assessment/

├── cubes/

│   ├── rgb/

│   │   ├── 0000.png

│   │   ├── 0001.png

│   │   └── ...

│   └── lidar/

│       ├── 0000.npy

│       ├── 0001.npy

│       └── ...

└── spheres/

    ├── rgb/

    └── lidar/
```

For Colab, upload data to Google Drive or use provided download scripts.


## Grading Rubric Details
### Task 1: W&B Experiment Tracking (20 points)

| Criterion | Points | Requirements |
|---|---:|---|
| Project setup | 4 | W&B project created and accessible |
| Metric logging | 8 | All required metrics logged correctly |
| Visualization logging | 2 | Loss curves in W&B |
| Hyperparameter config | 4 | All hyperparameters logged as config |

### Task 2: FiftyOne Visualization (15 points)

| Criterion | Points | Requirements |
|---|---:|---|
| Grouped dataset | 6 | Correct structure with RGB and LiDAR groups |
| Statistics | 4 | Complete dataset statistics documented |
| Exploration | 5 | Screenshots and written observations |

### Task 3: Fusion Comparison (20 points)

| Criterion | Points | Requirements |
|---|---:|---|
| Late fusion | 7.5 | Working implementation with results |
| Intermediate fusion | 7.5 | Working implementation with results |
| Analysis | 5 | Complete comparison table and written analysis |

### Task 4: Strided Convolution Ablation (20 points)

| Criterion | Points | Requirements |
|---|---:|---|
| Implementation | 8 | Correct EmbedderStrided class |
| Comparison | 6 | Complete metrics table for both variants |
| Analysis | 6 | Written analysis with justification |

### Task 5: CILP Performance (15 points)

| Criterion | Points | Requirements |
|---|---:|---|
| CILP loss < 3.5 | 5 | Validation loss threshold met |
| CILP loss < 3.2 | 2 | Bonus for lower loss |
| Projector MSE < 2.5 | 4 | Projector training successful |
| Accuracy > 95% | 4 | Final classifier meets threshold |
| Logging of similarity matrix | 2 | Logging of image-lidar similarity matrix as an artifact in W&B |

### Task 6: Code Quality (10 points)

| Criterion | Points | Requirements |
|---|---:|---|
| Organization | 4 | Modular structure, no duplication |
| Documentation | 3 | README, docstrings, comments |
| Reproducibility | 3 | Requirements file, seeds, instructions |