# X-IIoTID Results

Third benchmark dataset for ProtoIDS, run under the project's true open-set
protocol: withheld classes are removed **before** the scaler is fitted, no
prototypes are learned for them, and the UNKNOWN threshold is calibrated on
known validation rows only and then frozen for the test split.

## Dataset

- **File:** `X-IIoTID dataset.csv` (820,834 rows, 68 columns, 355,308,902 bytes)
- **SHA-256:** `7b9290057ee42e784da3c0d84b781815502c9205c74175c96374e71a5ffd98a0`
- **Source:** authors' archive of `munaalhawawreh/xiiotid-iiot-intrusion-dataset`
  (Kaggle, v1, 2021-10-08). This is the **68-column** release described in the
  paper, not the 42-column variant: the label columns are `class1/2/3`, not
  `Sub-Category/Attack Type`.
- **Label levels** (most → least granular): `class1` 19 classes, `class2` 10
  categories, `class3` binary. `class1` is the target.
- **Features retained:** 52 (of 68)
- **Split:** stratified 70/15/15, seed 42 → 535,075 train / 123,125 val / 123,126 test

## Columns removed, and why

| Column | Reason |
|---|---|
| `Date`, `Timestamp` | temporal markers leak capture-order |
| `Scr_IP`, `Des_IP` | identifiers |
| `Scr_port`, `Des_port` | port numbers |
| `class2`, `class3` | other label granularities |
| `anomaly_alert`, `OSSEC_alert`, `OSSEC_alert_level` | **third-party IDS verdicts** — not observed behaviour. Dropped by default because they let the model read another classifier's answer instead of learning from traffic. Kept as an explicit switch (`XIIOTID_EXCLUDE_IDS_ALERTS`) for ablation. |

The leakage filter tokenises column names rather than substring-matching.
Substring matching is unsafe here: `'ip'` occurs inside `Scr_ip_bytes` and
`'id'` inside `Avg_ideal_time`, so a naive filter silently deletes four real
measured features.

## Withheld classes (UNKNOWN)

`Scanning_vulnerability` (52,852) · `Dictionary` (2,572) · `Reverse_shell` (1,016)

Chosen to span three distinct attack categories — reconnaissance, credential
attack, exploitation — mirroring the CICIoT2023 selection rationale
(`VulnerabilityScan` + `DictionaryBruteForce` + a network-level attack). Classes
with fewer than ~1,000 rows (`MitM` 117, `Fake_notification` 28,
`crypto-ransomware` 458) were rejected as too rare for reliable statistics.
The benign class `Normal` is never withheld.

**Caveat:** this set was chosen by us, not validated independently. A second
withheld set is still needed to show the result is not an artefact of this choice.

## Reference run (documented default: K=3, λ=0.1, dim 32)

`experiments/results/xiiotid_base/`

| Metric | Value |
|---|---|
| Known accuracy | 0.9938 |
| Known macro-F1 | 0.8657 |
| Unknown F1 | 0.4327 |
| AUROC | 0.8629 |
| AUPR | 0.4570 |
| Threshold | 0.0481 |
| Unknown detection rate | 0.6510 |
| True FAR (unknown accepted as known) | 0.3490 |

## Swept result (best on validation, scored once on test)

**K=1, λ=0.1, embedding dim 64**, dropout 0.2, lr 1e-3, batch 256, 10 epochs,
seed 42. `experiments/results/sweep_xiiotid/sweep_coarse.json`

| Metric | Reference (K=3) | **Swept (K=1, dim 64)** | Δ |
|---|---|---|---|
| Known accuracy | 0.9938 | 0.9772 | −1.7 pp |
| Unknown F1 | 0.4327 | **0.4989** | **+6.6 pp** |
| AUROC | 0.8629 | **0.9245** | **+6.2 pp** |
| AUPR | 0.4570 | **0.5525** | **+9.6 pp** |
| Unknown detection rate | 0.6510 | 0.7834 | +13.2 pp |
| True FAR | 0.3490 | 0.2166 | −13.2 pp |
| Open-set FAR (known rejected) | 0.1003 | 0.1002 | — |
| Threshold | 0.0481 | 0.0602 | — |

The known-accuracy drop is the intended consequence of a more aggressive
threshold: 10% of known traffic is rejected to catch 13 pp more unknowns. The
selection objective weighted known accuracy 0.4 against unknown F1 0.4 and
AUROC 0.2, so this trade-off was tuned toward, not stumbled into. An operator
who needs higher known-accuracy should re-weight and re-select.

## K=1 beats K=3 — this contradicts the project default

Across the 18-trial grid (K ∈ {1,3,5} × λ ∈ {0,0.1,0.5} × dim ∈ {32,64}):

- best K=1 → objective 0.7722, AUROC 0.9217, AUPR 0.5387
- best K=5 → objective 0.7581, AUROC 0.9084, AUPR 0.5871
- best K=3 → objective 0.7266, AUROC 0.8430, AUPR 0.4029

`docs/RESEARCH_DECISIONS.md` justifies multiple prototypes on the grounds that
"K=3 improved unknown recall vs K=1" on CICIoT2023. That does **not** hold here.
More prototypes appear to fragment the class representation without helping
rejection. `RESEARCH_DECISIONS.md` should be updated before publication.

λ=0.5 is consistently the worst setting (6 of the 6 worst configs), so the
compactness term is not free — it needs a real budget to help.

## Comparison across the project's three datasets

| | CICIoT2023 | Edge-IIoTset | X-IIoTID (ref) | **X-IIoTID (swept)** |
|---|---|---|---|---|
| Known accuracy | 0.9897 | 0.9465 | 0.9938 | 0.9772 |
| Known macro-F1 | 0.7628 | 0.7072 | 0.8657 | — |
| Unknown F1 | 0.1114 | 0.4991 | 0.4327 | **0.4989** |
| AUROC | 0.9468 | 0.9281 | 0.8629 | **0.9245** |
| AUPR | 0.1246 | 0.3847 | 0.4570 | **0.5525** |

AUPR is the informative metric here because unknown traffic is ~7% of the test
split; the 0.7% class prior dominates AUROC, which is why a threshold-free
ranking can favour a dataset whose unknowns are simply more prevalent.

## Open items

- [ ] Threshold percentile sensitivity (80/85/90/95) — only p=90 evaluated
- [ ] Second withheld-class set, to show the result is not an artefact of ours
- [ ] Multiple seeds — everything above is seed 42
- [ ] Ablation: re-including the OSSEC/`anomaly_alert` IDS columns
- [ ] Cross-dataset validation (CICIoT2023 ↔ Edge-IIoTset ↔ X-IIoTID)

## Metric conventions (read before quoting these numbers)

`docs/RESEARCH_DECISIONS.md` defines **True FAR** as *the fraction of true
UNKNOWN samples incorrectly accepted as known* — the inverse of the standard
open-set FAR. `results/documented/CICIoT2023/open_set_test_results.json` also
labels its fields inconsistently: `unknown_precision` holds 0.0598, which is
actually the unknown **detection** rate, while `unknown_recall` 0.8209 does not
correspond to any count in that file. These names must be corrected before
anything is quoted in a report.
