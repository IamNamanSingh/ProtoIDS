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

## Swept result (best on validation, scored once on test) — **SUPERSEDED, see cross-validation below**

**K=1, λ=0.1, embedding dim 64**, dropout 0.2, lr 1e-3, batch 256, 10 epochs,
seed 42. `experiments/results/sweep_xiiotid/sweep_coarse.json`

| Metric | Reference (K=3) | Swept (K=1, dim 64) | Δ |
|---|---|---|---|
| Known accuracy | 0.9938 | 0.9772 | −1.7 pp |
| Unknown F1 | 0.4327 | 0.4989 | +6.6 pp |
| AUROC | 0.8629 | 0.9245 | +6.2 pp |
| AUPR | 0.4570 | 0.5525 | +9.6 pp |

> **These deltas are an artefact of the withheld-class set, not an
> improvement in the method.** This table was produced under protocol S1, the
> one withholding choice we made ourselves, and the K=1 advantage does not
> survive a change of protocol. The numbers above are kept only as the
> record of that single run. Read the cross-validation section instead.

## Cross-validation: is the result a property of the method or of our class choice?

Three independent open-set protocols, each with withheld classes completely
absent from training and the scaler fitted only on the remaining known-train
rows. Both configurations were run under every protocol so the K choice is
tested rather than assumed. Batch 256, 10 epochs, seed 42 throughout.

| protocol | withheld classes | cfg | known acc | unknown F1 | AUROC | AUPR |
|---|---|---|---|---|---|---|
| **S1** (ours) | Scanning_vulnerability, Dictionary, Reverse_shell | K=1 | 0.9772 | 0.4989 | 0.9245 | **0.5525** |
| | | K=3 | 0.9938 | 0.4327 | 0.8629 | 0.4570 |
| **S2** (disjoint) | Exfiltration, BruteForce, Modbus_register_reading | K=1 | 0.9948 | 0.3359 | 0.7512 | **0.3319** |
| | | K=3 | 0.9919 | 0.3498 | 0.8155 | 0.2950 |
| **F** (3 whole families) | C&C, Exfiltration, MQTT_cloud_broker_subscription, Modbus_register_reading, TCP Relay | K=1 | 0.9969 | 0.4806 | 0.9046 | **0.5681** |
| | | K=3 | 0.9963 | 0.4343 | 0.8319 | 0.4349 |

Protocol F is the most realistic zero-day test: the model has never seen *any*
member of those three attack categories, not merely a different type inside a
category it already knows.

### What this changes

**1. AUROC is dominated by the withholding choice, not by the model.**
Across protocols AUROC ranges **0.7512 – 0.9245**, a 17-point spread produced
purely by which classes were held out. Any single number quoted for this
dataset is meaningless without its protocol. AUPR is far more stable
(0.2950 – 0.5681) and is the metric to report.

**2. "K=1 beats K=3" is not established.** K=1 has the higher AUROC in S1
(+6.2 pp) and F (+7.3 pp) but is **worse in S2** (0.7512 vs 0.8155, −6.4 pp).
K=1 has the higher AUPR in all three protocols, so the advantage is real but
not universal. `docs/RESEARCH_DECISIONS.md` should be *qualified* on this
point, not reversed: multiple prototypes did not help on X-IIoTID, but
neither did fewer prototypes help unconditionally.

**3. The honest headline is the range, not the best cell.** Reporting
"AUROC 0.9245" from protocol S1 alone would misrepresent the method.

### What survives

- **AUPR is the trustworthy metric here**, and K=1/dim-64 leads on it in every
  protocol tested (0.5525 / 0.3319 / 0.5681 vs 0.4570 / 0.2950 / 0.4349).
- **The leakage-free pipeline is protocol-robust.** All three runs fit the
  scaler on exactly the known-train rows, and withheld classes never appear in
  the training split.
- **Family-level holdout (F) is the strongest protocol** and the one to report
  as the primary result: AUPR 0.5681, AUROC 0.9046, known accuracy 0.9969.

### Still not established

Three protocols is a small sample for characterising the spread. Before any of
this goes in a report, more withholding sets should be run to put an interval
on the variance, and multi-seed runs are needed since everything here is
seed 42.

## K sweep within protocol S1 (superseded — see cross-validation)

Across the 18-trial grid (K ∈ {1,3,5} × λ ∈ {0,0.1,0.5} × dim ∈ {32,64}), all
run under protocol S1 only:

- best K=1 → objective 0.7722, AUROC 0.9217, AUPR 0.5387
- best K=5 → objective 0.7581, AUROC 0.9084, AUPR 0.5871
- best K=3 → objective 0.7266, AUROC 0.8430, AUPR 0.4029

`docs/RESEARCH_DECISIONS.md` justifies multiple prototypes on the grounds that
"K=3 improved unknown recall vs K=1" on CICIoT2023. Within protocol S1 more
prototypes do not help, but the cross-validation above shows K=1 is not
uniformly better either, so this warrants a *qualification* of the existing
justification rather than its reversal.

λ=0.5 is consistently the worst setting (6 of the 6 worst configs), so the
compactness term is not free — it needs a real budget to help. This held across
the whole S1 grid.

## Comparison across the project's three datasets

X-IIoTID is reported as a **range over protocols**, because a single number
depends entirely on the withheld-class choice (see cross-validation above).

| | CICIoT2023 | Edge-IIoTset | X-IIoTID (ref, S1) | X-IIoTID (K=1, S1) | X-IIoTID (K=1, S2) | X-IIoTID (K=1, F) |
|---|---|---|---|---|---|---|
| Known accuracy | 0.9897 | 0.9465 | 0.9938 | 0.9772 | 0.9948 | 0.9969 |
| Known macro-F1 | 0.7628 | 0.7072 | 0.8657 | — | — | — |
| Unknown F1 | 0.1114 | 0.4991 | 0.4327 | 0.4989 | 0.3359 | 0.4806 |
| AUROC | 0.9468 | 0.9281 | 0.8629 | 0.9245 | 0.7512 | 0.9046 |
| AUPR | 0.1246 | 0.3847 | 0.4570 | 0.5525 | 0.3319 | 0.5681 |

AUPR is the informative metric because the unknown fraction differs per dataset
and per protocol, so a threshold-free ranking partly just measures how
prevalent the unknowns are. On AUPR, X-IIoTID is competitive with or better than
the other two datasets on every protocol except S2.

**Caveat on the other two columns:** the CICIoT2023 and Edge-IIoTset figures are
single runs under one withheld-class set each, and were produced before this
cross-validation was done. They are subject to exactly the same
protocol-dependence demonstrated here and should not be treated as
established numbers until re-run under matched protocols.

## Open items

- [x] Second withheld-class set (protocol S2) — done, see cross-validation
- [x] Whole-family holdout (protocol F) — done, the strongest protocol so far
- [ ] More withholding sets — 3 protocols is too few to put an interval on the
      17-point AUROC spread; this is now the highest-value remaining work
- [ ] Multiple seeds — everything here is seed 42
- [ ] Threshold percentile sensitivity (80/85/90/95) — only p=90 evaluated
- [ ] Re-run the hyper-parameter sweep *within* each protocol, since the S1
      sweep selected a config that S2 ranks worse than K=3
- [ ] Ablation: re-including the OSSEC/`anomaly_alert` IDS columns
- [ ] Re-validate CICIoT2023 and Edge-IIoTset under matched protocols before
      any cross-dataset comparison is quoted

## Metric conventions (read before quoting these numbers)

`docs/RESEARCH_DECISIONS.md` defines **True FAR** as *the fraction of true
UNKNOWN samples incorrectly accepted as known* — the inverse of the standard
open-set FAR. `results/documented/CICIoT2023/open_set_test_results.json` also
labels its fields inconsistently: `unknown_precision` holds 0.0598, which is
actually the unknown **detection** rate, while `unknown_recall` 0.8209 does not
correspond to any count in that file. These names must be corrected before
anything is quoted in a report.
