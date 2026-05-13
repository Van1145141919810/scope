"""Quick test: load pre-trained model and run forward pass."""
import sys; sys.path.insert(0, 'scripts')
import torch
from model import scope

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m = scope(1, 512, 1).to(device)
ckpt = torch.load('model/scope_model.pth', map_location=device)
m.load_state_dict(ckpt['model'])
m.eval()
print(f'Checkpoint loaded (epoch {ckpt["epoch"]})')

x = torch.randn(1, 10, 1, 64, 64).to(device)
pred, kl = m(x)
print(f'Forward pass OK: pred shape={pred.shape}, kl={kl.item():.4f}')
print('SCOPE environment ready!')
