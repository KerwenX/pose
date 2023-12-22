import torch

checkpoint = torch.load(
    '/home/aston/Desktop/python/HS-Pose/engine/output/models/remote/model_29.pth',
    map_location='cuda:0'
)

print(checkpoint['epoch'])
print(checkpoint.keys())

