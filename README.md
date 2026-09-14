# ProtoIDS

**Prototype-Based Open-Set IoT Intrusion Detection**

## What is ProtoIDS?
ProtoIDS is an experimental deep learning approach for IoT intrusion detection. Traditional classifiers simply draw boundaries between known attacks, which causes them to confidently misclassify unseen/zero-day attacks as "normal" or some known attack type. 

ProtoIDS solves this by learning "prototypes" (representative embeddings) for network behaviors. When an attack comes in, the model measures its cosine distance to these prototypes. If the attack is too far from any known prototype, it is flagged as **UNKNOWN**, rather than being misclassified.

## Project Status
**Current Phase: Baseline & Preprocessing Completed, ProtoIDS Architecture Implementation Pending**

What has been completed:
- Thorough forensic analysis and preprocessing of two massive datasets: CICIoT2023 and Edge-IIoTset.
- Created a manageable development subset (235,135 samples) of CICIoT2023 for rapid testing.
- Built and evaluated closed-set baselines (Random Forest, Gradient Boosting, MLP).

Current Development Baselines (CICIoT2023, Closed-Set):
- **Random Forest**: Accuracy ≈ 99.12%, Macro-F1 ≈ 82.40%
- **Gradient Boosting**: Accuracy ≈ 99.00%, Macro-F1 ≈ 74.63%
- **MLP**: Accuracy ≈ 91.09%, Macro-F1 ≈ 64.70%

*(Note: These are initial development results on known classes, not final research metrics. An older ablation study had scaling issues and its numbers should be ignored.)*

## Setup Instructions

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Datasets
The project uses two massive datasets which are **NOT** tracked in version control due to their size (multi-GB). You will need to download them and place them in the root directory:
- `CICIOT23/` (CICIoT2023 dataset)
- `DNN-EdgeIIoT-dataset.csv/` (Edge-IIoTset dataset)

*(Note: We already have preprocessed subsets in `experiments/data/` for rapid prototyping.)*

### 3. Running Existing Code
To run the preprocessing pipelines:
```bash
python src/preprocessing/ciciot_preprocessing.py
python src/preprocessing/edgeiiot_preprocessing.py
```

To run the baseline models on the development subset:
```bash
python src/baselines/run_baselines.py
```

## Next Steps
The next phase is to implement and test the ProtoIDS model (see `HANDOVER.md` for detailed developer instructions):
1. Fix the ablation study.
2. Implement the ProtoIDS encoder, prototype learning, and loss functions.
3. Conduct open-set experiments (withholding specific attack classes).
4. Evaluate cross-dataset generalization.
