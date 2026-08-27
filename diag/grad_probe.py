import sys, torch, torch.nn as nn, math
sys.path.insert(0, "/home/kmercad/mamba_har_2/project-basilisk")
from MambaSSL_JEPA_Model import MambaJEPA, HARMambaConfig
torch.manual_seed(0)
dev = torch.device("cuda")
config = HARMambaConfig()
lam = 0.1
model = MambaJEPA(config, mask_ratio=0.33, t_l=3, use_pe=False, drop=True, recon=True).to(dev)
crit = nn.SmoothL1Loss()
x = torch.randn(128, 90, 45, device=dev)

def stats(name, t):
    t = t.detach().float()
    print(f"  {name:28s} shape={tuple(t.shape)} mean={t.mean():.4f} std={t.std():.4f} absmax={t.abs().max():.4f} nan={torch.isnan(t).sum().item()} inf={torch.isinf(t).sum().item()}")

model.train()
te, targets, ctx_emb, mask_t, blocks, preds, ctx_recon = model(x)
print("=== FORWARD ===")
stats("target_embeddings", te); stats("targets", targets); stats("context_embeddings", ctx_emb)
stats("predictions", preds); stats("context_input_recon", ctx_recon)
print("  mask_t.sum per sample:", mask_t.sum(1)[:5].tolist(), "ctx_emb len", ctx_emb.shape[1])
jepa = crit(preds, targets)
recon = model.decoder(ctx_recon)
mask = mask_t.repeat_interleave(config.conv_stride, dim=1)
recon_loss = crit(recon[mask], x[mask])
print(f"jepa={jepa.item():.5f} recon={recon_loss.item():.5f}")
loss = lam*jepa + recon_loss

GROUPS = {
 "stem": list(model.context_encoder.convolutional_input.parameters())+list(model.context_encoder.LN_layer.parameters()),
 "input_proj": list(model.context_encoder.backbone.input_proj.parameters()),
 "early": [p for l in model.context_encoder.backbone.layers[0:3] for p in l.parameters()],
 "mid":   [p for l in model.context_encoder.backbone.layers[3:6] for p in l.parameters()],
 "late":  [p for l in model.context_encoder.backbone.layers[6:8] for p in l.parameters()]+list(model.context_encoder.backbone.norm_f.parameters()),
 "all":   list(model.context_encoder.parameters()),
 "predictor": list(model.predictor.parameters()),
 "decoder": list(model.decoder.parameters()),
}
print("=== autograd.grad per group (jepa / recon) ===")
for n, ps in GROUPS.items():
    ps = [p for p in ps if p.requires_grad]
    gj = torch.autograd.grad(jepa, ps, retain_graph=True, allow_unused=True)
    gr = torch.autograd.grad(recon_loss, ps, retain_graph=True, allow_unused=True)
    nj_none = sum(g is None for g in gj); nr_none = sum(g is None for g in gr)
    gj = torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(ps,gj)])
    gr = torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(ps,gr)])
    print(f"  {n:11s} nparams={len(ps):3d} |g_jepa|={gj.norm():.3e} (None:{nj_none}, nan:{torch.isnan(gj).sum().item()}, inf:{torch.isinf(gj).sum().item()}) |g_recon|={gr.norm():.3e} (None:{nr_none}, nan:{torch.isnan(gr).sum().item()}, inf:{torch.isinf(gr).sum().item()})")

print("=== full backward + clip ===")
model.zero_grad()
loss.backward()
tp = [p for p in model.parameters() if p.requires_grad]
norms = [(n, p.grad.norm().item() if p.grad is not None else None) for n,p in model.named_parameters() if p.requires_grad]
print("  params with None grad:", [n for n,v in norms if v is None][:10])
print("  params with nan/inf grad:", [n for n,v in norms if v is not None and not math.isfinite(v)][:20])
print("  zero-grad params:", [n for n,v in norms if v == 0.0][:20])
tot = torch.nn.utils.clip_grad_norm_(tp, max_norm=1.0)
print("  total_norm from clip:", tot.item())
print("  biggest grads:", sorted([(v,n) for n,v in norms if v is not None], reverse=True)[:8])
opt = torch.optim.AdamW(tp, lr=1.2e-4)
before = {n: p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
opt.step()
delta = {n: (p.detach()-before[n]).abs().max().item() for n,p in model.named_parameters() if p.requires_grad}
print("  max |delta param| after 1 step:", max(delta.values()), "  min:", max(0, min(delta.values())))
print("  any nan param:", any(torch.isnan(p).any().item() for p in model.parameters()))
