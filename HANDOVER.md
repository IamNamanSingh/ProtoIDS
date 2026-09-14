# ProtoIDS Developer Handover

Hey! Welcome to ProtoIDS. Here is the real talk on where the project stands and what you need to do next.

## Where we are now
We've finished the data wrangling. CICIoT2023 and Edge-IIoTset are massive, messy datasets. We did a full forensic analysis (see `reports/dataset_forensic_report.md`), figured out exactly which columns cause data leakage (timestamps, IPs, raw payloads), and wrote solid preprocessing scripts to clean them. We also ran standard ML baselines (Random Forest, GBM, MLP) on a development subset so we have something to compare our neural network against.

## What is already working
- **Preprocessing Pipelines:** `src/preprocessing/` has working scripts for both datasets. They handle scaling, NaN filling, and dropping leakage columns.
- **Baselines:** `src/baselines/run_baselines.py` is working and gives us our baseline metrics on a ~235k row development subset.
- **Dev Data:** `experiments/data/ciciot_dev.parquet` is our curated development subset. Use this for quick testing so you don't have to load 5.5 million rows every time.

## What should NOT be trusted yet
- **The old ablation results:** The numbers in the previous ablation experiment are weird (like 2% accuracy). There was a scaling or label mapping bug in that specific script. Do not trust those old ablation numbers; the script needs fixing.
- **The ProtoIDS implementation:** The code in `src/protoids/` is a work in progress. It's scaffolded but hasn't been fully trained or tested yet.

## What you should do next
1. **Fix the ablation script:** Figure out why the quick ablation script tanked the accuracy and re-run it properly.
2. **Implement ProtoIDS:** Finish the architecture in `src/protoids/`. See "How the ProtoIDS model is supposed to work" below.
3. **Closed-set tests:** Train ProtoIDS on all known classes and compare to the Random Forest baseline.
4. **Open-set experiments:** This is the core research. Train the model but withhold `MITM-ArpSpoofing`, `VulnerabilityScan`, and `DictionaryBruteForce`. Then test if the model correctly flags them as UNKNOWN instead of blindly guessing a known class.
5. **Hyperparameter testing:** Test K=1, K=3, and K=5 prototypes per class.
6. **Cross-dataset validation:** Validate on at least 3 benchmark datasets (we have CICIoT and Edge-IIoT ready).
7. **Compare against existing methods:** Compare the results with suitable state-of-the-art approaches.
8. **Final reporting and demo:** Perform proper ablation/analysis, prepare the final technical report, and build the frontend/demo later.

## Important Research Rules
1. **The goal is NOT just high accuracy.** We want open-set detection. A model with 99% accuracy that fails to detect zero-day attacks is useless to us.
2. **Threshold Calibration:** The threshold for UNKNOWN detection MUST be calibrated using only known-class validation data. Never use unknown test labels to tune your threshold.
3. **No Leakage:** Keep IP addresses, ports, and timestamps out of the features.
4. **Class Imbalance:** Class imbalance will be investigated using appropriate methods such as class-weighting or balanced sampling.

## Important Files
- `reports/dataset_forensic_report.md`: Read this. It explains the datasets and leakage risks.
- `reports/protoids_methodology.md`: The math and architecture behind the model.
- `src/protoids/`: Where you'll be writing most of your code.
- `src/preprocessing/ciciot_preprocessing.py`: How we clean the primary dataset.
- `src/baselines/ablation_quick.py`: The broken ablation script you need to fix first.

## How the ProtoIDS model is supposed to work
The architecture is simple but specific:
1. **Encoder:** Input → Linear(128) → BatchNorm → ReLU → Dropout(0.2) → Linear(64) → ReLU → Linear(32) → L2 normalization.
2. **Prototypes:** We learn multiple prototypes per class (initially using KMeans on the embeddings, then they become learnable parameters).
3. **Distance:** We use **Cosine Distance**.
4. **Classification:** The predicted class is the closest prototype. If the distance to the closest prototype is greater than a calibrated threshold, the model outputs UNKNOWN.
