import torch
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os

class MRIDataset(Dataset):
    def __init__(self, image_dir):
        self.image_path = os.path.join(image_dir, "sample_mri.png")
        self.mask_path = os.path.join(image_dir, "sample_mask.png")
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image = Image.open(self.image_path).convert("RGB").resize((128, 128))
        mask = Image.open(self.mask_path).resize((128, 128))
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs["labels"] = torch.tensor(np.array(mask)[None, :, :], dtype=torch.long)
        return inputs

dataset = MRIDataset("data")
loader = DataLoader(dataset, batch_size=1)

model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

for epoch in range(1):  # Only 1 epoch
    for batch in loader:
        optimizer.zero_grad()
        outputs = model(**{k: v.squeeze(0) for k, v in batch.items()})
        loss = outputs.loss
        print("Loss:", loss.item())
        loss.backward()
        optimizer.step()
