"""Pretrain AE on CICIoT dev (MSE) then save encoder for ProtoIDS init"""
import torch, torch.nn as nn, argparse
from .dataset import create_data_loaders
from .autoencoder import ProtoIDSAutoEncoder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--embedding_dim", type=int, default=32)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _, _, input_dim = create_data_loaders("CICIOT23", args.batch_size)
    model = ProtoIDSAutoEncoder(input_dim, args.embedding_dim).to(device)
    print(f"AutoEncoder Architecture:")
    print(f"  - Encoder: {input_dim} -> 128 -> BatchNorm -> ReLU -> Dropout(0.2) -> 64 -> ReLU -> {args.embedding_dim} -> L2 norm")
    print(f"  - Decoder: {args.embedding_dim} -> 64 -> BatchNorm -> ReLU -> 128 -> BatchNorm -> ReLU -> {input_dim}")
    print(f"  - Loss: MSE reconstruction")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for ep in range(args.epochs):
        model.train()
        tl=0; n=0
        for X,_ in train_loader:
            X=X.to(device)
            recon,_,_=model(X)
            loss=loss_fn(recon, X)
            opt.zero_grad(); loss.backward(); opt.step()
            tl+=loss.item()*len(X); n+=len(X)
        # val
        model.eval(); vl=0; vn=0
        with torch.no_grad():
            for X,_ in val_loader:
                X=X.to(device)
                recon,_,_=model(X)
                vl+=loss_fn(recon, X).item()*len(X); vn+=len(X)
        print(f"Epoch {ep} train MSE {tl/n:.4f} val MSE {vl/vn:.4f}")
    torch.save({"encoder_state": model.encoder.state_dict(), "cfg": {"input_dim": input_dim, "embedding_dim": args.embedding_dim}}, "experiments/models/ae_pretrain.pth")
    print("Saved to experiments/models/ae_pretrain.pth - use to init ProtoIDS encoder before KMeans")

if __name__ == "__main__":
    main()
