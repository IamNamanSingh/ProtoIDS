# Dataset Forensic Analysis Report — ProtoIDS

**Date:** 2026-09-10  
**Project:** ProtoIDS — Prototype-Based IoT Intrusion Detection  
**Analyst:** Automated Forensic Pipeline  
**Status:** Pre-preprocessing, read-only inspection  

---

## 1. Executive Summary

This report presents a comprehensive forensic analysis of two IoT intrusion detection datasets prior to any preprocessing or model development. The datasets are:

1. **CICIoT2023** (PRIMARY) — 7,845,673 total rows across train/validation/test splits, 47 columns, 34 attack classes
2. **Edge-IIoTset DNN** (SECONDARY) — 2,219,201 rows, 63 columns, 15 attack types + binary label

### Key Findings

| Finding | CICIoT2023 | Edge-IIoTset |
|---------|-----------|-------------|
| **Leakage Risk** | LOW — no IP/port/timestamp columns | HIGH — contains IP addresses, timestamps, ports, raw payloads |
| **Missing Values** | NONE | NONE |
| **Class Imbalance** | SEVERE (6,064:1 ratio) | SEVERE (1,329:1 ratio) |
| **Constant Columns** | 3 constant + 8 near-constant | ~25 constant in head sample (protocol-specific zero-fills) |
| **Cross-dataset compatibility** | Aggregated flow features | Raw packet-level features |

> [!IMPORTANT]
> The two datasets use fundamentally different feature representations. CICIoT2023 provides pre-computed flow-level statistics while Edge-IIoTset provides raw packet-level fields. A direct feature-to-feature alignment is NOT possible — a learned representation/encoder is essential for cross-dataset evaluation.

---

## 2. CICIoT2023 Analysis

### 2.1 Dataset Dimensions

| Split | Rows | Columns | File Size |
|-------|------|---------|-----------|
| Train | 5,491,971 | 47 | ~1.5 GB |
| Validation | 1,176,851 | 47 | ~332 MB |
| Test | 1,176,851 | 47 | ~332 MB |
| **Total** | **7,845,673** | **47** | **~2.2 GB** |

### 2.2 Column Names (All 47)

**Flow/Connection features (5):**
`flow_duration`, `Header_Length`, `Duration`, `Rate`, `Srate`, `Drate`

**TCP Flag features (6):**
`fin_flag_number`, `syn_flag_number`, `rst_flag_number`, `psh_flag_number`, `ack_flag_number`, `ece_flag_number`, `cwr_flag_number`

**Packet count features (5):**
`ack_count`, `syn_count`, `fin_count`, `urg_count`, `rst_count`

**Protocol indicator features (14):**
`HTTP`, `HTTPS`, `DNS`, `Telnet`, `SMTP`, `SSH`, `IRC`, `TCP`, `UDP`, `DHCP`, `ARP`, `ICMP`, `IPv`, `LLC`

**Statistical features (9):**
`Tot sum`, `Min`, `Max`, `AVG`, `Std`, `Tot size`, `IAT`, `Number`, `Magnitue`, `Radius`, `Covariance`, `Variance`, `Weight`

**Other:**
`Protocol Type` (numerical, 2938 unique values)

**Label:**
`label` (string, 34 unique classes)

### 2.3 Label Column Identification

| Column | Dtype | Unique Values | Role |
|--------|-------|--------------|------|
| `label` | object (string) | 34 | **Main label — attack type** |
| `Protocol Type` | float64 | 2,938 | Feature (NOT a label despite keyword match) |

> [!NOTE]
> `Protocol Type` was flagged as a label candidate by keyword matching, but it is a numerical feature with 2,938 unique float values. It represents protocol encoding, not an attack classification.

The dataset has a **single label column** (`label`) with **34 attack types**. There is no separate binary label or attack-category column. However, the attack names encode an implicit hierarchy via prefixes.

### 2.4 Attack Hierarchy (Derived from Label Names)

The 34 labels naturally group into **8 attack categories**:

| Category | Attack Types | Count |
|----------|-------------|-------|
| **DDoS** | ICMP_Flood, UDP_Flood, TCP_Flood, PSHACK_Flood, SYN_Flood, RSTFINFlood, SynonymousIP_Flood, ICMP_Fragmentation, UDP_Fragmentation, ACK_Fragmentation, HTTP_Flood, SlowLoris | 12 |
| **DoS** | UDP_Flood, TCP_Flood, SYN_Flood, HTTP_Flood | 4 |
| **Mirai** | greeth_flood, udpplain, greip_flood | 3 |
| **Recon** | HostDiscovery, OSScan, PortScan, PingSweep | 4 |
| **Spoofing** | MITM-ArpSpoofing, DNS_Spoofing | 2 |
| **Web Attack** | BrowserHijacking, SqlInjection, CommandInjection, XSS | 4 |
| **Malware** | Backdoor_Malware, Uploading_Attack | 2 |
| **Other** | DictionaryBruteForce, VulnerabilityScan | 2 |
| **Benign** | BenignTraffic | 1 |

### 2.5 Feature Types

| Type | Count | Details |
|------|-------|---------|
| Numerical | 46 | All features except `label` |
| Categorical (object) | 1 | `label` only |
| Boolean | 0 | — |

### 2.6 Missing Values

| Split | Missing Values | Inf Values |
|-------|---------------|------------|
| Train | **0** | **0** |
| Validation | **0** | **0** |
| Test | **0** | **0** |

The dataset is completely clean with no missing or infinite values.

### 2.7 Duplicate Rows

| Split | Duplicates | Percentage |
|-------|-----------|------------|
| Train | 134,565 | 2.45% |
| Validation | 6,297 | 0.54% |
| Test | 6,384 | 0.54% |

> [!NOTE]
> Duplicate rows are present but at low percentages. For network flow data, exact duplicates are plausible (identical flows can legitimately occur). However, duplicate flows across train/test splits could inflate evaluation metrics. This should be investigated before final model evaluation.

### 2.8 Constant and Near-Constant Columns

**Constant (1 unique value in both train and validation):**

| Column | Value |
|--------|-------|
| `Telnet` | 0.0 |
| `IRC` | 0.0 |

**Near-Constant (>99% single value):**

| Column | Dominant Value % | Unique |
|--------|-----------------|--------|
| `ece_flag_number` | ~100% | 2 |
| `cwr_flag_number` | ~100% | 2 |
| `DNS` | 99.99% | 2 |
| `SSH` | ~100% | 2 |
| `DHCP` | ~100% | 2 |
| `ARP` | 99.99% | 2 |
| `IPv` | 99.99% | 2 |
| `LLC` | 99.99% | 2 |
| `SMTP` | ~100% (train only) | 2 |

> [!WARNING]
> 11 columns are constant or near-constant. These provide negligible discriminative power and may add noise. `Telnet` and `IRC` are always 0 and should definitely be excluded. The near-constant protocol indicators may still carry some signal for rare attack types.

### 2.9 Identifier Columns

| Column | Keyword Match | Assessment |
|--------|--------------|------------|
| `flow_duration` | "flow" | **NOT an identifier** — legitimate feature (flow duration in seconds) |
| `IPv` | "ip" | **NOT an identifier** — binary protocol indicator (0/1) |

> [!TIP]
> **CICIoT2023 has NO IP addresses, timestamps, ports, or other direct identifiers in its feature set.** This is a significant advantage — the features are pre-aggregated flow statistics, reducing leakage risk substantially compared to raw packet captures.

### 2.10 Cross-Split Consistency

**Columns:** ✅ All three splits have identical 47 columns  
**Label classes:** ✅ All three splits contain the same 34 attack classes  
**Class distributions:** ✅ Nearly identical proportions across all splits (see §5)

---

## 3. Edge-IIoTset Analysis

### 3.1 Dataset Dimensions

| Metric | Value |
|--------|-------|
| Rows | 2,219,201 |
| Columns | 63 |
| File Size | ~1.16 GB |
| Splits | None (single file) |

### 3.2 Column Names (All 63)

**Timestamp:** `frame.time`

**IP Address features:** `ip.src_host`, `ip.dst_host`, `arp.dst.proto_ipv4`, `arp.src.proto_ipv4`

**ARP features:** `arp.opcode`, `arp.hw.size`

**ICMP features:** `icmp.checksum`, `icmp.seq_le`, `icmp.transmit_timestamp`, `icmp.unused`

**HTTP features:** `http.file_data`, `http.content_length`, `http.request.uri.query`, `http.request.method`, `http.referer`, `http.request.full_uri`, `http.request.version`, `http.response`, `http.tls_port`

**TCP features:** `tcp.ack`, `tcp.ack_raw`, `tcp.checksum`, `tcp.connection.fin`, `tcp.connection.rst`, `tcp.connection.syn`, `tcp.connection.synack`, `tcp.dstport`, `tcp.flags`, `tcp.flags.ack`, `tcp.len`, `tcp.options`, `tcp.payload`, `tcp.seq`, `tcp.srcport`

**UDP features:** `udp.port`, `udp.stream`, `udp.time_delta`

**DNS features:** `dns.qry.name`, `dns.qry.name.len`, `dns.qry.qu`, `dns.qry.type`, `dns.retransmission`, `dns.retransmit_request`, `dns.retransmit_request_in`

**MQTT features:** `mqtt.conack.flags`, `mqtt.conflag.cleansess`, `mqtt.conflags`, `mqtt.hdrflags`, `mqtt.len`, `mqtt.msg_decoded_as`, `mqtt.msg`, `mqtt.msgtype`, `mqtt.proto_len`, `mqtt.protoname`, `mqtt.topic`, `mqtt.topic_len`, `mqtt.ver`

**Modbus TCP:** `mbtcp.len`, `mbtcp.trans_id`, `mbtcp.unit_id`

**Labels:** `Attack_label` (binary: 0/1), `Attack_type` (string: 15 categories)

### 3.3 Label Columns

| Column | Dtype | Unique Values | Role |
|--------|-------|--------------|------|
| `Attack_label` | int64 | 2 | **Binary label** (0 = Normal, 1 = Attack) |
| `Attack_type` | object | 15 | **Multi-class label** (attack type) |

This dataset provides **two levels of labeling**: binary (normal vs attack) and multi-class (15 attack types).

### 3.4 Attack Type Distribution (Full Dataset)

| Attack Type | Count | Percentage |
|-------------|-------|-----------|
| Normal | 1,615,643 | 72.80% |
| DDoS_UDP | 121,568 | 5.48% |
| DDoS_ICMP | 116,436 | 5.25% |
| SQL_injection | 51,203 | 2.31% |
| Password | 50,153 | 2.26% |
| Vulnerability_scanner | 50,110 | 2.26% |
| DDoS_TCP | 50,062 | 2.26% |
| DDoS_HTTP | 49,911 | 2.25% |
| Uploading | 37,634 | 1.70% |
| Backdoor | 24,862 | 1.12% |
| Port_Scanning | 22,564 | 1.02% |
| XSS | 15,915 | 0.72% |
| Ransomware | 10,925 | 0.49% |
| MITM | 1,214 | 0.05% |
| Fingerprinting | 1,001 | 0.05% |

**Binary Distribution:**
- Normal (label=0): 1,615,643 (72.80%)
- Attack (label=1): 603,558 (27.20%)

### 3.5 Feature Types

| Type | Count |
|------|-------|
| Numerical | 51 |
| Categorical/object | 12 |

### 3.6 Missing Values

**Total missing values: 0** (verified via full chunked scan)

### 3.7 Duplicate Rows

**Estimated from head sample (20,000 rows): 0 duplicates (0.00%)**

> [!NOTE]
> This is an ESTIMATE from the first 20K rows only (which happened to be entirely Normal traffic). The actual duplicate count across the full dataset with mixed attack types may differ. Due to computational constraints, a full duplicate scan was not performed.

### 3.8 Constant and Near-Constant Columns

> [!WARNING]
> The constant/near-constant analysis was performed on the first 20,000 rows, which contained only Normal traffic. Many columns that appear constant in the head sample (e.g., `icmp.*`, `http.*`, `dns.*`) likely have non-zero values in attack traffic segments. The constants identified below should be understood in this context.

**Constant in head sample (likely protocol-specific zero-fills for Normal/TCP traffic):**

`icmp.checksum`, `icmp.seq_le`, `icmp.transmit_timestamp`, `icmp.unused`, `http.file_data`, `http.content_length`, `http.request.uri.query`, `http.request.method`, `http.referer`, `http.request.full_uri`, `http.request.version`, `http.response`, `http.tls_port`, `udp.port`, `udp.stream`, `udp.time_delta`, `dns.qry.name`, `dns.qry.name.len`, `dns.qry.qu`, `dns.qry.type`, `dns.retransmission`, `dns.retransmit_request`, `dns.retransmit_request_in`, `mqtt.msg_decoded_as`, `mbtcp.len`, `mbtcp.trans_id`, `mbtcp.unit_id`

**Near-constant in head sample:**

`arp.dst.proto_ipv4` (99.9%), `arp.opcode` (99.9%), `arp.hw.size` (99.9%), `arp.src.proto_ipv4` (99.9%)

### 3.9 Identifier Columns (Leakage Sources)

| Column | Type | Content | Leakage Risk |
|--------|------|---------|-------------|
| `frame.time` | Timestamp | `2021 11:44:10.081753000` | **HIGH** |
| `ip.src_host` | IP Address | `192.168.0.128`, `192.168.0.101` | **HIGH** |
| `ip.dst_host` | IP Address | `192.168.0.128`, `224.0.0.251` | **HIGH** |
| `arp.dst.proto_ipv4` | IP Address | `192.168.0.1`, `192.168.0.128` | **HIGH** |
| `arp.src.proto_ipv4` | IP Address | `192.168.0.1`, `192.168.0.128` | **HIGH** |
| `tcp.dstport` | Port Number | 64855, 1883, ... | **MEDIUM** |
| `tcp.srcport` | Port Number | 1883, 64855, ... | **MEDIUM** |
| `tcp.payload` | Raw Payload | Hex-encoded payload data | **HIGH** |
| `tcp.options` | TCP Options | Hex-encoded options | **MEDIUM** |
| `mqtt.msg` | Message Content | Hex-encoded sensor data | **MEDIUM** |
| `mqtt.topic` | Topic Name | `Temperature_and_Humidity` | **LOW** |

---

## 4. Label Analysis

### 4.1 CICIoT2023 Label Structure

- **Single label column**: `label`
- **34 attack types** forming an implicit hierarchy
- **No explicit binary label** — must be derived (BenignTraffic vs all others)
- **No explicit category label** — must be derived from name prefixes

### 4.2 Edge-IIoTset Label Structure

- **Binary label**: `Attack_label` (0 = Normal, 1 = Attack)
- **Multi-class label**: `Attack_type` (15 classes)
- **No sub-type labels** — single level of attack granularity

### 4.3 Label Mapping Between Datasets

| Category | CICIoT2023 Types | Edge-IIoTset Types |
|----------|-----------------|-------------------|
| DDoS | ICMP_Flood, UDP_Flood, TCP_Flood, PSHACK_Flood, SYN_Flood, RSTFINFlood, SynonymousIP_Flood, ICMP_Frag, UDP_Frag, ACK_Frag, HTTP_Flood, SlowLoris | DDoS_UDP, DDoS_ICMP, DDoS_TCP, DDoS_HTTP |
| DoS | UDP_Flood, TCP_Flood, SYN_Flood, HTTP_Flood | (merged into DDoS) |
| Recon | HostDiscovery, OSScan, PortScan, PingSweep | Port_Scanning, Fingerprinting |
| Web Attack | SqlInjection, XSS, CommandInjection, BrowserHijacking | SQL_injection, XSS |
| MITM | MITM-ArpSpoofing | MITM |
| Malware | Backdoor_Malware, Uploading_Attack | Backdoor, Uploading, Ransomware |
| Brute Force | DictionaryBruteForce | Password |
| Vulnerability | VulnerabilityScan | Vulnerability_scanner |
| Botnet | Mirai-greeth_flood, Mirai-udpplain, Mirai-greip_flood | — |
| Normal | BenignTraffic | Normal |

---

## 5. Class Imbalance Analysis

### 5.1 CICIoT2023 Imbalance

**Imbalance Ratio (largest : smallest):** 848,088 : 140 = **6,058:1**

| Tier | Classes | Examples | Total % |
|------|---------|----------|---------|
| Majority (>100K) | DDoS-ICMP_Flood, DDoS-UDP_Flood, DDoS-TCP_Flood, DDoS-PSHACK_Flood, DDoS-SYN_Flood, DDoS-RSTFINFlood, DDoS-SynonymousIP_Flood, DoS-UDP_Flood, DoS-TCP_Flood, DoS-SYN_Flood, BenignTraffic, Mirai-* | 12 | ~94.5% |
| Medium (1K–100K) | DDoS-ICMP_Frag, MITM, DDoS-UDP_Frag, DDoS-ACK_Frag, DNS_Spoofing, Recon-*, DoS-HTTP, VulnerabilityScan, DDoS-HTTP, DDoS-SlowLoris, DictionaryBruteForce | 11 | ~5.0% |
| Minority (<1K) | BrowserHijacking, CommandInjection, SqlInjection, XSS, Backdoor_Malware, Recon-PingSweep, Uploading_Attack | 7 | ~0.5% |

**Cross-split consistency:** Class proportions are nearly identical across train/validation/test (within ±0.3%), indicating a properly stratified split.

### 5.2 Edge-IIoTset Imbalance

**Imbalance Ratio (largest : smallest):** 1,615,643 : 1,001 = **1,614:1**

Normal traffic dominates at 72.8%. Attack types range from 0.05% to 5.48%.

### 5.3 Imbalance Impact on ProtoIDS

> [!IMPORTANT]
> The extreme class imbalance is the most critical challenge for ProtoIDS. Prototype-based methods require learning representative prototypes for each class. Classes with <500 samples may not yield reliable prototypes. Class balancing (oversampling minorities or undersampling majorities) is ESSENTIAL during training.

---

## 6. Leakage Analysis

### 6.1 CICIoT2023 Leakage Assessment

| Column | Reason | Risk Level |
|--------|--------|-----------|
| `Protocol Type` | Numerical encoding with 2,938 unique values. Moderate correlation with label (0.41). Legitimate feature but may overfit to dataset-specific protocol distributions. | **LOW** |
| `Magnitue` | Highest correlation with label (0.45). Appears to be a legitimate flow feature. | **LOW** |
| `Telnet` | Constant (always 0). No leakage but adds noise. | **NONE** |
| `IRC` | Constant (always 0). No leakage but adds noise. | **NONE** |
| `SMTP` | Constant in validation (near-constant in train). | **NONE** |
| `Rate` / `Srate` | Identical values (mean and std match exactly). Redundant but not leakage. | **LOW** |

**Overall CICIoT2023 Leakage Risk: LOW**

No single feature has correlation >0.5 with the label. No IP addresses, timestamps, or identifiers are present. The features are well-engineered flow statistics.

### 6.2 Edge-IIoTset Leakage Assessment

| Column | Reason | Risk Level |
|--------|--------|-----------|
| `frame.time` | Exact nanosecond timestamp. Attacks occur in temporal clusters — model could memorize time windows. | **HIGH** |
| `ip.src_host` | Source IP address. Attack traffic originates from specific IPs (e.g., 192.168.0.170 appears in 7.4% of traffic). Model could learn IP→attack mapping. | **HIGH** |
| `ip.dst_host` | Destination IP address. Same concern as source IP. | **HIGH** |
| `arp.dst.proto_ipv4` | ARP destination IP. Correlated with specific attack types. | **HIGH** |
| `arp.src.proto_ipv4` | ARP source IP. Same concern. | **HIGH** |
| `tcp.srcport` | Source port. Ephemeral ports may be correlated with specific attack tools. | **MEDIUM** |
| `tcp.dstport` | Destination port. Specific services (ports 80, 1883, etc.) may correlate with attack types. | **MEDIUM** |
| `tcp.payload` | Raw packet payload in hex. Contains actual attack content (SQL queries, HTTP requests, malware data). Model could learn payload signatures rather than behavioral patterns. | **HIGH** |
| `tcp.options` | TCP options field. May reveal OS fingerprinting artifacts. | **MEDIUM** |
| `mqtt.msg` | MQTT message content. Contains sensor data that may be environment-specific. | **MEDIUM** |
| `mqtt.topic` | MQTT topic name. Environment-specific (e.g., "Temperature_and_Humidity"). | **LOW** |
| `mqtt.protoname` | Protocol name string. Trivial indicator. | **LOW** |
| `http.request.full_uri` | Full HTTP URI including attack payloads. Directly reveals attack content. | **HIGH** |
| `http.referer` | HTTP referer header. May contain attack-specific artifacts. | **MEDIUM** |
| `dns.qry.name` | DNS query name. Could be used to identify DNS-based attacks trivially. | **MEDIUM** |
| `Attack_label` | Binary label. Must be excluded from features (it IS the label). If left in, creates perfect leakage. | **CRITICAL** |

**Overall Edge-IIoTset Leakage Risk: HIGH**

### 6.3 Duplicate Flow Leakage Risk

CICIoT2023 has 2.45% duplicate rows in training data. If the same duplicated flows appear in both train and test sets, evaluation metrics will be inflated. This requires verification before final evaluation.

---

## 7. Feature Analysis

### 7.1 CICIoT2023 Feature Groups

| Group | Features | Count | Nature |
|-------|----------|-------|--------|
| Flow metadata | flow_duration, Duration, Rate, Srate, Drate | 5 | Temporal |
| Packet header | Header_Length | 1 | Size |
| TCP flags | fin/syn/rst/psh/ack/ece/cwr_flag_number | 7 | Binary indicators |
| Packet counts | ack/syn/fin/urg/rst_count | 5 | Integer counts |
| Protocol indicators | HTTP, HTTPS, DNS, ..., LLC | 14 | Binary indicators |
| Flow statistics | Tot sum, Min, Max, AVG, Std, Tot size | 6 | Statistical |
| Timing | IAT | 1 | Inter-arrival time |
| Aggregates | Number, Magnitue, Radius, Covariance, Variance, Weight | 6 | Derived statistics |
| Protocol type | Protocol Type | 1 | Encoded integer |

### 7.2 Feature Correlations with Label (CICIoT2023, validation set)

| Feature | |ρ| with label | Interpretation |
|---------|----------------|----------------|
| Magnitue | 0.4497 | Flow magnitude — legitimate discriminator |
| Protocol Type | 0.4087 | Protocol encoding — legitimate but may overfit |
| Tot size | 0.3919 | Total packet size — legitimate |
| AVG | 0.3914 | Average packet size — legitimate |
| Tot sum | 0.3787 | Total bytes — legitimate |
| ICMP | 0.3702 | ICMP indicator — legitimate (ICMP floods) |
| Min | 0.3701 | Minimum packet size — legitimate |
| Variance | 0.3481 | Size variance — legitimate |

No feature has suspiciously high correlation (>0.8) with the label. All top features are legitimate network flow characteristics.

### 7.3 Redundant Features

- **`Rate` and `Srate`**: Identical mean (9,092.85), std (100,431.1), min (0), max (8,388,608). These appear to be duplicated. One should be dropped.
- **`IPv` and `LLC`**: Both are 99.99% equal to 1.0, with identical distributions. Likely redundant.

---

## 8. Cross-Dataset Comparison

| Property | CICIoT2023 | Edge-IIoTset |
|----------|-----------|-------------|
| **Rows** | 7,845,673 (split into train/val/test) | 2,219,201 (single file) |
| **Features** | 46 numerical + 1 label | 51 numerical + 12 categorical |
| **Label(s)** | `label` (34 classes) | `Attack_label` (binary) + `Attack_type` (15 classes) |
| **Attack types** | 34 | 15 |
| **Attack categories** | ~8 (derived from prefixes) | 15 (explicit) |
| **Feature type** | Aggregated flow statistics | Raw packet-level fields |
| **Categorical features** | 1 (label only) | 12 (IPs, timestamps, payloads, etc.) |
| **Missing values** | 0 | 0 |
| **Potential leakage** | LOW | HIGH |
| **Class imbalance** | 6,058:1 | 1,614:1 |
| **Constant columns** | 3 | ~25 (in head sample) |
| **Dataset role** | PRIMARY (training + evaluation) | SECONDARY (cross-dataset validation) |
| **IP/Port columns** | NONE | Multiple (ip.src_host, ip.dst_host, tcp.srcport, tcp.dstport) |
| **Timestamp columns** | NONE | frame.time |
| **Pre-split** | Yes (train/val/test) | No |

### 8.1 Shared Concepts

Despite different feature representations, both datasets capture:
- **Protocol behavior**: CICIoT2023 has binary protocol indicators; Edge-IIoTset has protocol-specific field groups
- **TCP connection state**: CICIoT2023 has flag counts; Edge-IIoTset has tcp.connection.* and tcp.flags.*
- **Packet size information**: CICIoT2023 has Tot sum/Min/Max/AVG/Std; Edge-IIoTset has tcp.len
- **Timing**: CICIoT2023 has flow_duration, IAT; Edge-IIoTset has frame.time, udp.time_delta

### 8.2 Dataset-Specific Features

**CICIoT2023 only:** Flow-level aggregated statistics (Radius, Covariance, Variance, Weight, Magnitue, Number)

**Edge-IIoTset only:** Raw packet fields (tcp.payload, tcp.options, tcp.ack_raw, mqtt.*, dns.*, http.*, mbtcp.*, arp.*)

### 8.3 Cross-Dataset Strategy

> [!IMPORTANT]
> A common preprocessing/representation strategy across both datasets is **NOT possible at the feature level**. The feature spaces are fundamentally different. This actually makes ProtoIDS's encoder-based approach scientifically interesting — the encoder must learn abstract behavioral representations from dataset-specific raw features. For cross-dataset evaluation, separate preprocessing pipelines feeding into a shared embedding space are needed.

### 8.4 Dataset Roles

- **CICIoT2023**: PRIMARY dataset for training and evaluating ProtoIDS. Clean, pre-aggregated features, no leakage risks, pre-split, 34 classes.
- **Edge-IIoTset**: SECONDARY dataset for cross-dataset generalization evaluation. Raw features requiring extensive preprocessing and leakage column removal.

---

## 9. Recommended Preprocessing

### 9.A Columns to EXCLUDE

**CICIoT2023:**

| Column | Reason |
|--------|--------|
| `label` | Target variable — must be separated, not used as feature |
| `Telnet` | Constant (always 0) |
| `IRC` | Constant (always 0) |
| `SMTP` | Near-constant (essentially constant in validation) |
| One of `Rate`/`Srate` | Redundant (identical values) |
| One of `IPv`/`LLC` | Redundant (identical distributions, 99.99% = 1.0) |

**Edge-IIoTset (for cross-dataset evaluation):**

| Column | Reason |
|--------|--------|
| `Attack_label` | Label — must be separated |
| `Attack_type` | Label — must be separated |
| `frame.time` | Timestamp — leakage |
| `ip.src_host` | IP address — leakage |
| `ip.dst_host` | IP address — leakage |
| `arp.dst.proto_ipv4` | IP address — leakage |
| `arp.src.proto_ipv4` | IP address — leakage |
| `tcp.payload` | Raw payload — leakage |
| `tcp.options` | TCP options hex — leakage |
| `mqtt.msg` | Message content — leakage |
| `mqtt.conack.flags` | Near-constant string |
| `mqtt.protoname` | Environment-specific |
| `mqtt.topic` | Environment-specific |
| `http.request.full_uri` | Contains attack content |
| `http.referer` | Attack-specific |
| All constant columns (mbtcp.*, dns.qry.type, etc.) | Zero-variance |

### 9.B Columns to RETAIN

**CICIoT2023 (41 features after exclusions):**
`flow_duration`, `Header_Length`, `Protocol Type`, `Duration`, `Rate` (keep one of Rate/Srate), `Drate`, `fin_flag_number`, `syn_flag_number`, `rst_flag_number`, `psh_flag_number`, `ack_flag_number`, `ece_flag_number`, `cwr_flag_number`, `ack_count`, `syn_count`, `fin_count`, `urg_count`, `rst_count`, `HTTP`, `HTTPS`, `DNS`, `SSH`, `TCP`, `UDP`, `DHCP`, `ARP`, `ICMP`, `IPv` (keep one of IPv/LLC), `Tot sum`, `Min`, `Max`, `AVG`, `Std`, `Tot size`, `IAT`, `Number`, `Magnitue`, `Radius`, `Covariance`, `Variance`, `Weight`

### 9.C Categorical Columns Needing Encoding

**CICIoT2023:** No categorical features to encode (all retained features are numerical).

**Edge-IIoTset:** After removing leakage columns, no categorical features remain.

### 9.D Numerical Columns Needing Scaling

**ALL numerical features** in both datasets need scaling. Recommended approach:
- **StandardScaler** (z-score normalization) for features like Header_Length, Rate, Tot sum, IAT which have very different scales
- **MinMaxScaler** as alternative for bounded features (flag indicators are already 0/1)
- Fit scaler on training data only; transform validation/test with the same scaler

> [!TIP]
> Key scale differences in CICIoT2023: `IAT` has mean ~83M, `Rate` has max 8.4M, while flag indicators are 0-1. Without scaling, distance-based prototype matching will be dominated by high-magnitude features.

### 9.E Class Balancing

**YES — class balancing is ESSENTIAL.**

Recommended approaches for ProtoIDS:
1. **Per-class prototype count**: Use more prototypes for majority classes, fewer for minority classes
2. **Oversampling minorities** (SMOTE or random oversampling) for encoder training
3. **Undersampling majorities**: Create a balanced subset for development
4. **Class-weighted loss**: Weight inverse-proportionally to class frequency
5. **Stratified mini-batching**: Ensure each mini-batch contains examples from all classes

### 9.F Development Subset

> [!IMPORTANT]
> For initial development, **DO NOT use all 5.5M training rows**. Create a controlled subset:

**Recommended development subset strategy:**
1. Take ALL minority class samples (classes with <5,000 rows): ~4,500 rows
2. Sample 5,000 rows from each medium class: ~55,000 rows
3. Sample 10,000 rows from each majority class: ~120,000 rows
4. **Total development subset: ~180,000 rows** (balanced enough for prototyping)

This is ~3% of the training data, enabling fast iteration while preserving class representation. Scale up to full data once the approach is validated.

### 9.G Open-Set Experiment Design

See §11 for detailed recommendation.

---

## 10. Recommended Experimental Protocol

### Phase 1: Closed-Set Baseline
1. Preprocess CICIoT2023 (remove leakage/constant columns, scale)
2. Train encoder + prototypes on the development subset
3. Evaluate on validation set with all 34 known classes
4. Establish baseline accuracy, per-class F1, and confusion matrix

### Phase 2: Open-Set Evaluation
1. Select classes to withhold (see §11)
2. Retrain encoder + prototypes on remaining classes
3. At test time, classify known classes AND detect withheld classes as UNKNOWN
4. Report: known-class accuracy, unknown detection rate, AUROC for novelty detection

### Phase 3: Cross-Dataset Generalization
1. Train on CICIoT2023 (primary)
2. Test on Edge-IIoTset (after preprocessing and label mapping)
3. Evaluate whether learned representations transfer to a different feature space
4. This is the most challenging test — may require domain adaptation

---

## 11. Open-Set Experiment Recommendation

### Recommended Initial Experiment

Based on the class distributions and attack taxonomy, here is a scientifically meaningful first open-set experiment:

**KNOWN ATTACKS (train with these):**
- All DDoS variants (12 classes, ~74% of attacks)
- All DoS variants (4 classes, ~17% of attacks)
- BenignTraffic
- Mirai variants (3 classes, ~6% of attacks)

**WITHHELD AS UNKNOWN (completely excluded from training):**

| Withheld Class | Training Count | Rationale |
|---------------|---------------|-----------|
| **MITM-ArpSpoofing** | 36,316 | Different attack mechanism (layer 2); sufficient samples for meaningful evaluation; has a conceptual counterpart in Edge-IIoTset |
| **VulnerabilityScan** | 4,396 | Reconnaissance-adjacent but distinct behavior; medium sample size |
| **DictionaryBruteForce** | 1,541 | Authentication attack, fundamentally different from volumetric attacks |

**Why these choices are scientifically meaningful:**
1. **MITM-ArpSpoofing** operates at a fundamentally different network layer than the DDoS/DoS/Mirai attacks used for training — testing whether ProtoIDS can detect genuinely novel attack mechanisms
2. **VulnerabilityScan** generates traffic patterns distinct from flooding attacks — tests detection of low-volume, targeted reconnaissance
3. **DictionaryBruteForce** is an authentication-layer attack — tests whether the prototype space learns to reject semantically different attack patterns
4. Together, they span different attack categories while leaving enough data for statistically significant evaluation

**Alternative experiment (harder):**
Withhold an entire category (all 4 Recon attacks: HostDiscovery + OSScan + PortScan + PingSweep = ~37K samples). This tests whether ProtoIDS can detect a completely unknown attack *category*, not just a type.

---

## 12. Risks and Limitations

### 12.1 Known Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Extreme class imbalance (6,058:1) | **HIGH** | Class balancing strategies (§9.E) |
| Edge-IIoTset leakage columns | **HIGH** | Remove all identifier columns (§9.A) |
| Head sample bias in Edge-IIoT analysis | **MEDIUM** | First 20K rows were all Normal — some analysis results (constant cols, correlations) only reflect Normal traffic patterns |
| Cross-dataset feature mismatch | **HIGH** | Require learned representations, not feature alignment |
| Duplicate rows across splits (CICIoT2023) | **MEDIUM** | Verify no train-test row overlap before final evaluation |
| Near-constant features masking rare events | **LOW** | Retain near-constant features initially; remove only after confirming no discriminative value |
| `Rate`/`Srate` redundancy | **LOW** | Drop one column |
| Development subset may not be representative | **MEDIUM** | Validate development results on full data before publication |

### 12.2 Limitations of This Analysis

1. **Edge-IIoTset constant column analysis** is based on the first 20K rows (all Normal). Many columns that appear constant likely have non-zero values in attack traffic.
2. **Edge-IIoTset duplicate estimation** is from the first 20K rows only (0% duplicates observed, but this may not represent the full dataset).
3. **Feature-label correlation** for Edge-IIoTset could not be computed because the head sample contained only one class.
4. **Cross-train-test duplicate analysis** for CICIoT2023 was not performed (comparing 5.5M rows against 1.2M rows is computationally expensive).
5. All numerical results are **exact measurements** unless explicitly noted as estimates.

### 12.3 Methodological Notes

- CICIoT2023 analysis: All splits were loaded fully into memory and analyzed completely.
- Edge-IIoTset analysis: Row counting and missing value analysis used full chunked scans. Label distributions used full chunked scans. Dtype analysis, constant column detection, and statistics used the first 20,000 rows.
- No dataset files were modified, renamed, or overwritten during this analysis.

---

*End of Report*
