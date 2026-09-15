"""Cross-dataset evaluation: CICIoT-trained ProtoIDS on Edge-IIoTset."""
import os, json, argparse, numpy as np, torch, joblib, pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from .protoids_model import ProtoIDS

# Map Edge-IIoT Attack_type -> CICIoT label where possible (forensic §4.3)
EDGE_TO_CICIOT = {
    "Normal": "BenignTraffic",
    "DDoS_UDP": "DDoS-UDP_Flood",
    "DDoS_ICMP": "DDoS-ICMP_Flood",
    "DDoS_TCP": "DDoS-TCP_Flood",
    "DDoS_HTTP": "DDoS-HTTP_Flood",
    "Port_Scanning": "Recon-PortScan",
    "Vulnerability_scanner": "VulnerabilityScan",
    "SQL_injection": "SqlInjection",
    "XSS": "XSS",
    "Backdoor": "Backdoor_Malware",
    "Uploading": "Uploading_Attack",
    "Password": "DictionaryBruteForce",
    "MITM": "MITM-ArpSpoofing",
    "Ransomware": None,  # no direct CICIoT counterpart
    "Fingerprinting": "Recon-OSScan",
}

def load_edge_sample(n=50000):
    df = pd.read_csv("DNN-EdgeIIoT-dataset.csv/DNN-EdgeIIoT-dataset.csv", low_memory=False, nrows=n)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.fillna(0)
    with open("results/baselines/edgeiiot_feature_manifest.json") as f:
        mani = json.load(f)
    cols = mani["retained_columns"]
    scaler = joblib.load("results/baselines/edgeiiot_preprocessor.joblib")
    X = scaler.transform(df[cols].values)
    X = np.clip(X, -5, 5).astype(np.float32)
    y_edge = df["Attack_type"].values
    return X, y_edge, cols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="experiments/models/protoids_k3_clip/model.pth")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=20000)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading model {args.model_path} on {device}")
    ckpt = torch.load(args.model_path, map_location=device)
    cfg = ckpt["protoids_args"]
    model = ProtoIDS(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    with open("results/baselines/ciciot_label_mapping.json") as f:
        lm = json.load(f)
    label_to_int = lm["multiclass"]["label_to_int"]
    int_to_label = {v:k for k,v in label_to_int.items()}

    X_edge, y_edge, cols = load_edge_sample(args.n)
    # Input dim mismatch check: CICIoT 41 vs Edge 41 (coincidentally same) but semantics differ
    print(f"Edge sample {X_edge.shape}, CICIoT input_dim {cfg['input_dim']}, Edge cols {len(cols)}")
    if X_edge.shape[1] != cfg["input_dim"]:
        print("DIM MISMATCH – cannot share encoder directly (forensic §8 notes different feature spaces).")
        print("Need separate encoder or domain adapt. Aborting naive transfer.")
        return

    # Map edge labels to CICIoT int where possible, otherwise mark -1 (ignore)
    y_mapped = []
    valid_mask = []
    for lab in y_edge:
        ciciot = EDGE_TO_CICIOT.get(lab, None)
        if ciciot and ciciot in label_to_int:
            y_mapped.append(label_to_int[ciciot])
            valid_mask.append(True)
        else:
            y_mapped.append(-1)
            valid_mask.append(False)
    y_mapped = np.array(y_mapped)
    valid_mask = np.array(valid_mask)
    print(f"Mappable {valid_mask.sum()}/{len(valid_mask)} ({valid_mask.mean():.2%}) – Ransomware etc. excluded")

    X_valid = torch.from_numpy(X_edge[valid_mask]).to(device)
    y_valid = torch.from_numpy(y_mapped[valid_mask]).to(device)

    with torch.no_grad():
        _, norm, _, min_dist_per_class = model(X_valid)
        pred = torch.argmin(min_dist_per_class, dim=1).cpu().numpy()
        y_valid_np = y_valid.cpu().numpy()
        acc = accuracy_score(y_valid_np, pred)
        macro = f1_score(y_valid_np, pred, average="macro", zero_division=0)
        print(f"Cross-dataset (CICIoT→Edge, naive 41→41 transfer, n={valid_mask.sum()}): acc {acc:.4f} macroF1 {macro:.4f}")
        print("Note: low acc expected – feature spaces are fundamentally different (forensic §8.1), true cross-dataset needs domain adaptation, not raw transfer.")

if __name__ == "__main__":
    main()
