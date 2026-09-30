# Research Decisions and Justifications

This document captures key research decisions made during the ProtoIDS project, along with their justifications and tradeoffs.

## Dataset Selection and Withholding Strategy

### Decision: CICIoT2023 as Primary Dataset
**Justification**:
- Large-scale, recent IoT dataset with diverse attack types
- Provides standard train/validation/test splits
- Contains sufficient samples for meaningful open-set evaluation
- Well-documented in literature, facilitating comparison

### Decision: Edge-IIoTset as Secondary Dataset
**Justification**:
- Represents a different IoT environment (industrial IoT)
- Enables cross-dataset validation and leakage analysis
- Helps assess generalization capabilities of the model
- Smaller size allows for quicker experimentation cycles

### Decision: Withholding Specific Classes for Open-Set Evaluation
**CICIoT2023**:
- Withheld: MITM-ArpSpoofing, VulnerabilityScan, DictionaryBruteForce
- **Justification**: 
  - Represent distinct attack categories (network-level, vulnerability scanning, credential-based)
  - Likely to appear as zero-day attacks in real-world scenarios
  - Have sufficient samples in validation/test for meaningful statistics
  - Not overly rare (avoiding extreme imbalance issues)

**Edge-IIoTset**:
- Withheld: MITM, Password, Vulnerability_scanner
- **Justification**:
  - Cover different attack vectors in industrial IoT context
  - Represent clinically relevant zero-day scenarios
  - Chosen based on feature distribution analysis to ensure they are sufficiently distinct

## Architectural Decisions

### Decision: Prototype-Based Approach
**Justification**:
- Provides interpretable representation of known classes
- Naturally supports distance-based unknown detection
- More flexible than fixed centroids (learnable prototypes)
- Allows multiple prototypes per class to capture multimodal distributions

### Decision: Multiple Prototypes per Class (K>1)
**Justification**:
- Many attack classes exhibit multiple behavioral modes (e.g., different variants or staging)
- Single prototype may not adequately represent complex class distributions
- Empirical ablation showed improvements in unknown recall with K=3 vs K=1
- Tradeoff: Increased parameters and potential overfitting with very high K

### Decision: Cosine Distance Metric
**Justification**:
- Embedding is L2-normalized, making cosine distance equivalent to Euclidean distance up to scale
- Cosine distance is invariant to embedding magnitude, focusing on direction
- Naturally bounded between 0 and 2, facilitating threshold interpretation
- Alternative metrics (e.g., Euclidean) tested but showed similar performance with cosine being slightly more stable

### Decision: L2 Normalization of Embeddings
**Justification**:
- Essential for cosine distance to be meaningful
- Prevents embedding scale from dominating distance calculations
- Helps prevent prototype vectors from growing arbitrarily large
- Common practice in metric learning and prototype-based methods

### Decision: Temperature Scaling (Not Used)
**Considered**: Applying temperature scaling to logits or distances
**Decision**: Not implemented
**Justification**:
- Initial experiments showed minimal improvement on validation set
- Added complexity without clear benefit
- Kept the model simpler and more interpretable
- Can be revisited in future work if needed

## Loss Function and Training Decisions

### Decision: Combined Loss (Cross-Entropy + Compactness)
**Formulation**: 
   L = L_ce + λ * L_compact
   where L_compact encourages samples to stay close to their class prototypes
**Justification**:
- Cross-entropy alone does not encourage compact clustering of embeddings
- Compactness loss improves intra-class cohesion, aiding separation
- Ablation study showed lambda=0.1 provided best balance of known-class and unknown-class performance
- Lambda=0 (no compactness) decreased unknown recall significantly despite improving known-class accuracy

### Decision: Adam Optimizer
**Justification**:
- Adaptive learning rate helps navigate complex loss landscapes
- Reliable performance across different architectures and datasets
- Default choice for deep learning experiments unless specific constraints exist
- Alternative optimizers (SGD with momentum) tested but showed slower convergence

### Decision: Learning Rate Schedule (Fixed)
**Considered**: Learning rate decay or cosine annealing
**Decision**: Fixed learning rate (0.001)
**Justification**:
- Preliminary experiments showed stable convergence with fixed LR
- Added complexity of schedules did not yield significant improvement
- Kept training configuration simple for reproducibility
- Can be explored in future optimization work

### Decision: Dropout Rate (0.2)
**Justification**:
- Standard dropout rate for preventing overfitting in fully connected networks
- Ablation with 0.1 and 0.3 showed 0.2 provided best validation performance
- Balances regularization with model capacity
- Consistent with similar architectures in literature

## Evaluation Protocol Decisions

### Decision: Threshold Calculation Method
**Method**: 90th percentile of minimum prototype distance on known validation samples
**Justification**:
- Balances known-class accuracy and unknown-class detection
- Empirically tested on validation set; 90th percentile provided reasonable tradeoff
- Alternative methods (fixed threshold, other percentiles) evaluated in sensitivity studies
- Using known validation samples only prevents data leakage from unknown classes
- Threshold is then frozen and used for test set evaluation (never recalibrated on test)

### Decision: True FAR Convention
**Definition**: True FAR = fraction of true unknown samples incorrectly accepted as known
**Justification**:
- Aligns with standard binary classification false positive rate for the unknown class
- Consistent with open-set literature where unknown detection is treated as a binary task
- Avoids confusion with other FAR definitions (e.g., false alarm rate per known class)
- Clearly documented to prevent misinterpretation in results

### Decision: Reporting Known vs Unknown Metrics Separately
**Justification**:
- Overall accuracy can be misleading in open-set scenarios (high due to known class dominance)
- Known-class metrics reflect performance on training data
- Unknown-class metrics reflect the core open-set detection capability
- Reporting both provides complete picture of model behavior
- Enables calculation of metrics like KRR (Known Rejection Rate) for complementarity

## Experimental Design Decisions

### Decision: Single Runs with Fixed Seed
**Justification**:
- Resource constraints limited extensive statistical studies
- Fixed seed (42) ensures reproducibility of reported results
- Ablation studies and sensitivity analyses conducted with same seed for fair comparison
- Future work should include multiple runs with different seeds for statistical significance

### Decision: Development Subset for Initial Iteration
**Justification**:
- Full dataset processing is computationally expensive and memory-intensive
- Development subset (~235k rows) allows rapid hypothesis testing
- Results on development subset trends consistent with full dataset (where available)
- Final validation on full dataset required for publication-quality results

### Decision: Cross-Dataset Validation Approach
**Justification**:
- Train on one dataset, test on another (and vice versa)
- Measures true generalization capability beyond memorization
- Requires careful feature alignment and preprocessing consistency
- Planned for future work after individual dataset optimization

## Limitations and Future Work Avoidance

### Decision: Not Pursuing State-of-the-Art Comparison Prematurely
**Justification**:
- Primary focus is on establishing a sound open-set detection protocol
- Comparing to SOTA requires significant additional work (reimplementing baselines)
- Will be addressed after core ProtoIDS optimization and validation
- Current limitations (memory efficiency, third dataset) take precedence

### Decision: Not Implementing Complex Uncertainty Quantification
**Justification**:
- ProtoIDS provides a clear UNKNOWN/KKNOWN decision based on distance threshold
- Additional uncertainty estimates (e.g., Bayesian) add complexity
- Core contribution is the prototype-based open-set framework
- Can be extended in future work if needed for specific applications

### Decision: Avoiding Over-Optimization on Single Metric
**Justification**:
- Open-set performance involves tradeoffs between known-class and unknown-class metrics
- Optimizing solely for unknown recall may degrade known-class usability
- Decisions based on balanced evaluation (e.g., F1 scores, AUC)
- Explicitly documented tradeoffs in ablation studies (K and lambda variations)

---
*Last updated: $(date +%Y-%m-%d)*