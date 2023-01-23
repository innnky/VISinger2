import torch

ckpt_path = "/Volumes/Extend/下载/G_493000.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")

ckpt['optimizer'] = None
torch.save(ckpt, ckpt_path)
