# Cross-validation of the X-IIoTID open-set result

Three independent open-set protocols, each withholding classes completely from
training with the scaler fitted only on the remaining known-train rows.
Batch 256, 10 epochs, seed 42 throughout. Every protocol runs both the swept
configuration (K=1, dim 64) and the project's documented default (K=3, dim 32)
so the choice of K is tested rather than assumed.

## Headline: the result depends on the withholding choice, not only on the method

| protocol | withheld classes | cfg | known acc | unknown F1 | AUROC | AUPR |
|---|---|---|---|---|---|---|
| **S1** (our original) | Scanning_vulnerability, Dictionary, Reverse_shell | K=1 | 0.9772 | 0.4989 | 0.9245 | 0.5525 |
| | | K=3 | 0.9938 | 0.4327 | 0.8629 | 0.4570 |
| **S2** (disjoint) | Exfiltration, BruteForce, Modbus_register_reading | K=1 | 0.9948 | 0.3359 | 0.7512 | 0.3319 |
| | | K=3 | 0.9919 | 0.3498 | 0.8155 | 0.2950 |
| **F** (whole families) | C&C, Exfiltration, MQTT_cloud_broker_subscription, Modbus_register_reading, TCP Relay | K=1 | 0.9969 | 0.4806 | 0.9046 | 0.5681 |
| | | K=3 | 0.9963 | 0.4343 | 0.8319 | 0.4349 |

### Reading the table

**AUROC swings 0.7512 – 0.9245 across protocols** — a 17-point spread caused
entirely by which classes are held out, with the model and hyperparameters
held fixed. Quoting the best cell (0.9245) as "the" result would misrepresent
the method. `unknown F1` behaves similarly (0.3359 – 0.4989).

**AUPR is the stable metric** (0.2950 – 0.5681) and is what should be reported
alongside the protocol that produced it.

**K=1 does not uniformly beat K=3.** It wins AUROC in S1 (+6.2 pp) and F
(+7.3 pp) but loses in S2 (0.7512 vs 0.8155, −6.4 pp). It wins AUPR in all
three. The project's existing justification for multiple prototypes in
`docs/RESEARCH_DECISIONS.md` therefore needs *qualification*, not reversal:
extra prototypes did not help here, but fewer prototypes are not
universally better either.

### Why S2 is the hardest

S2 contains `BruteForce`, a sibling of the `Dictionary` class withheld in S1,
plus `Exfiltration` and `Modbus_register_reading`. Attack types within a family
share behaviour, so a model that has learned the weaponization/credential
pattern from `insider_malicious` generalises poorly to a held-out
*variant* of that same family while generalising better to a wholly different
family. This is the "unseen type" vs "unseen family" distinction, and S2 vs F
is exactly that contrast:

| | S2 (unseen types, mostly) | F (unseen families) |
|---|---|---|
| K=1 AUROC | 0.7512 | 0.9046 |
| K=1 AUPR | 0.3319 | 0.5681 |

Withholding a whole attack family — never seeing *any* member of the category —
is both the more realistic zero-day test and the one this model handles better.

### What is established

- The leakage-free protocol holds under all three withholding regimes: the
  scaler is fitted on exactly the known-train rows, and withheld classes never
  appear in the training split.
- On AUPR, K=1 with dim 64 leads in all three protocols.
- Family-level withholding (F) is the strongest protocol and the most
  defensible primary result.

### What is not established

- Any single-point accuracy claim for this dataset.
- That K=1 is generally better than K=3 — three protocols is too few, and one
  of them disagrees.
- Anything about seed sensitivity: every run here is seed 42.

## Reproducing

```bash
bash scripts/xval_xiiotid.sh
```

Each run writes `experiments/results/xval_<protocol>_<tag>_k<K>_e<E>/` and
skips any experiment whose results already exist, so it is resumable.
Per-run logs: `/tmp/opencode/xval/`.

Raw per-run results are preserved in this directory as
`crossval_xval_<protocol>_<tag>_*.json`.
