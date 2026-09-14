# ProtoIDS Methodology: Prototype-Based Open-Set IoT Intrusion Detection

## Overview

ProtoIDS (Prototype-based Intrusion Detection System) is a deep learning approach for IoT intrusion detection that learns prototypical representations of network behaviors in an embedding space. Unlike traditional classification methods that learn decision boundaries, ProtoIDS learns class prototypes and assigns samples to the class of their nearest prototype, with a distance-based threshold for detecting unknown attacks.

## Mathematical Formulation

### Input and Embedding

Let $x \in \mathbb{R}^d$ be a network flow feature vector after preprocessing, where $d$ is the number of input features.

The encoder function $f_\theta: \mathbb{R}^d \rightarrow \mathbb{R}^m$ maps the input to an embedding space:
$$z = f_\theta(x)$$

where $z \in \mathbb{R}^m$ is the raw embedding before normalization, and $m$ is the embedding dimension.

### L2 Normalization

To ensure meaningful distance comparisons, we apply L2 normalization to the embedding:
$$\hat{z} = \frac{z}{\|z\|_2}$$

where $\hat{z}$ is the L2-normalized embedding lying on the unit hypersphere.

### Protoype Representation

For each class $c \in \{0, 1, ..., C-1\}$ where $C$ is the number of known classes, we learn $K$ prototypes:
$$P_c = \{p_{c,1}, p_{c,2}, ..., p_{c,K}\} \subset \mathbb{R}^m$$

where each prototype $p_{c,k}$ is also L2-normalized to lie on the unit hypersphere.

### Distance Calculation

We use cosine distance as our similarity metric. For a normalized embedding $\hat{z}$ and a normalized prototype $p$, the cosine distance is:
$$d(\hat{z}, p) = 1 - \cos(\hat{z}, p) = 1 - \hat{z}^\top p$$

Since both vectors are L2-normalized, their dot product equals the cosine of the angle between them.

### Class Distance

For each class $c$, we define the distance to the nearest prototype:
$$D_c(\hat{z}) = \min_{k=1..K} d(\hat{z}, p_{c,k})$$

### Classification Decision

In closed-set mode (known classes only), the predicted class is:
$$c^* = \arg\min_{c=0..C-1} D_c(\hat{z})$$

In open-set mode, we introduce a threshold $\tau$ for unknown detection:
$$\text{prediction} = 
\begin{cases}
c^* & \text{if } \min_{c} D_c(\hat{z}) \leq \tau \\
\text{UNKNOWN} & \text{if } \min_{c} D_c(\hat{z}) > \tau
\end{cases}$$

### Training Objective

The ProtoIDS model is trained with a composite loss function:
$$\mathcal{L} = \mathcal{L}_{classification} + \lambda \mathcal{L}_{compactness}$$

where $\lambda$ is a hyperparameter controlling the importance of the compactness term.

#### Classification Loss

We convert distances to similarities for cross-entropy loss:
$$s_c = -D_c(\hat{z})$$

Then apply standard cross-entropy:
$$\mathcal{L}_{classification} = -\log \frac{\exp(s_{y})}{\sum_{c=0}^{C-1} \exp(s_c)}$$

where $y$ is the true class label.

#### Compactness Loss

The compactness loss encourages embeddings to be close to their true class's nearest prototype:
$$\mathcal{L}_{compactness} = \frac{1}{N} \sum_{i=1}^{N} D_{y_i}(\hat{z}_i)$$

where $N$ is the batch size, $y_i$ is the true label of sample $i$, and $\hat{z}_i$ is its normalized embedding.

This loss minimizes the average distance between samples and their class's nearest prototype, encouraging compact clusters around prototypes.

## Architecture Details

### Encoder Network

The encoder follows the specified architecture:
- Input → Linear(128) → BatchNorm → ReLU → Dropout(0.2) → Linear(64) → ReLU → Linear(32) → L2 normalization

This maps input features to a 32-dimensional embedding space where intrusion detection decisions are made.

### Prototype Learning

Protoypes are initialized using K-means clustering on the training embeddings:
1. Train encoder for a few epochs to get initial embeddings
2. Apply K-means (K=3 per class) to initialize prototype positions
3. Treat prototypes as learnable parameters and jointly optimize with encoder

During training, both encoder parameters and prototype vectors are updated via backpropagation.

## Key Design Choices

### Why Cosine Distance?

Cosine distance is invariant to the magnitude of vectors, focusing only on direction. This is beneficial because:
1. It emphasizes pattern similarity over magnitude
2. Works well with L2-normalized embeddings
3. Naturally handles varying scales in different feature dimensions

### Why Multiple Prototypes per Class?

Network attacks within the same class can exhibit significant variation (e.g., different variants of malware). Multiple prototypes allow the model to capture:
- Sub-types or variants within an attack class
- Different operational modes of the same attack
- Improved coverage of the class manifold in embedding space

### Why L2 Normalization?

L2 normalization ensures that:
1. Distance metrics are not dominated by large-magnitude features
2. All embeddings lie on a comparable hypersphere
3. Cosine distance becomes a meaningful angular distance measure
4. Prevents the encoder from trivially increasing embedding magnitudes to reduce loss

## Threshold Calibration

For open-set detection, the threshold $\tau$ is calibrated using only known-class validation data:
1. Compute minimum distances $D_c(\hat{z})$ for all known validation samples
2. Set $\tau$ as the $p$-th percentile of these distances (typically p=90)
3. This ensures that a known percentage (e.g., 90%) of known samples are accepted
4. The threshold represents the maximum distance at which we're confident a sample belongs to a known class

## Advantages Over Baseline Methods

Compared to traditional machine learning baselines (Random Forest, Gradient Boosting, MLP):

1. **Interpretability**: Decisions are based on proximity to learned prototypes, which can be inspected
2. **Open-set Capability**: Naturally extends to detect unknown attacks via distance thresholding
3. **Feature Space Learning**: Learns an informative embedding space rather than working in raw feature space
4. **Class Separation**: Explicitly encourages compact clusters around prototypes while separating classes
5. **Flexibility**: Number of prototypes per class can be adjusted based on class complexity

## Limitations

1. **Prototype Initialization Quality**: Depends on the quality of initial embeddings for K-means
2. **Threshold Sensitivity**: Performance depends on proper threshold calibration
3. **Embedding Dimension**: Requires careful selection of embedding dimension
4. **Computational Cost**: Training neural networks is more expensive than tree-based methods

## Implementation Notes

- All prototypes are L2-normalized during distance computation to ensure cosine distance validity
- The encoder uses batch normalization for stable training
- Dropout is applied to prevent overfitting
- Class weighting can be incorporated to handle severe class imbalance
- Early stopping based on validation loss prevents overfitting

## Flow of Information

```
Raw Network Features [x]
        ↓
Preprocessing (StandardScaler, feature selection)
        ↓
Encoder Network f_θ
        ↓
Raw Embedding [z]
        ↓
L2 Normalization → [ẑ]
        ↓
Distance to Prototypes: d(ẑ, p) = 1 - ẑᵀp
        ↓
Nearest Prototype per Class: D_c(ẑ) = min_k d(ẑ, p_{c,k})
        ↓
Classification: c* = argmin_c D_c(ẑ)
        ↓
(open-set) Threshold Check: if min_c D_c(ẑ) > τ → UNKNOWN
```

This methodology provides a principled approach to learning discriminative representations for IoT intrusion detection with inherent open-set capabilities.