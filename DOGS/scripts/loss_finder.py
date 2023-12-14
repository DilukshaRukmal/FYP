#Script to find loss in each saved checkpoint
import os
import torch

save_dir = "/home/diluksha/FYP_OOD/Sajeepan/DOGS/checkpoints"
for i in range(10):
    if (i + 1) % 2 == 0:
        model_checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{i+1}.pt")

        # Load the checkpoint
        checkpoint = torch.load(model_checkpoint_path)

        # Access the loss from the checkpoint dictionary
        loss = checkpoint["loss"]

        print(f"Loss in checkpoint: {loss}")

