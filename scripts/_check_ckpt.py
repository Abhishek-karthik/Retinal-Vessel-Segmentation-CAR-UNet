import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
ckpt = torch.load("outputs/saved_models/car_unet_fives.pt", map_location="cpu")
print("Checkpoint OK")
print("Epoch:", ckpt["epoch"])
print("Best Val Dice:", round(ckpt["best_val_dice"], 4))
print("History epochs logged:", len(ckpt["history"]["train_loss"]))
