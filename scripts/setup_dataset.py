import os
import glob

def setup_dataset():
    """
    Verifies the DRIVE dataset structure provided in data/DRIVE/
    
    Structure:
      training/
        ├── images/      (20 fundus images, 21_training.tif to 40_training.tif)
        ├── 1st_manual/  (20 ground-truth vessel masks, 21_manual1.gif to 40_manual1.gif)
        └── mask/        (20 FOV masks, 21_training_mask.gif to 40_training_mask.gif)
      test/
        ├── images/      (20 fundus images, 01_test.tif to 20_test.tif)
        └── mask/        (20 FOV masks, 01_test_mask.gif to 20_test_mask.gif)
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "DRIVE"))
    
    train_images = glob.glob(os.path.join(base_dir, "training", "images", "*.tif"))
    train_manuals = glob.glob(os.path.join(base_dir, "training", "1st_manual", "*.gif"))
    train_masks = glob.glob(os.path.join(base_dir, "training", "mask", "*.gif"))
    
    test_images = glob.glob(os.path.join(base_dir, "test", "images", "*.tif"))
    test_masks = glob.glob(os.path.join(base_dir, "test", "mask", "*.gif"))
    
    print("=" * 60)
    print("DRIVE DATASET INTEGRITY VERIFICATION")
    print("=" * 60)
    print(f"Base Directory: {base_dir}")
    print(f"Training Set  : {len(train_images)} Images | {len(train_manuals)} Ground Truth Masks | {len(train_masks)} FOV Masks")
    print(f"Held-out Test : {len(test_images)} Images | {len(test_masks)} FOV Masks")
    print("=" * 60)
    
    assert len(train_images) == 20, f"Expected 20 training images, found {len(train_images)}"
    assert len(train_manuals) == 20, f"Expected 20 training ground truth masks, found {len(train_manuals)}"
    assert len(train_masks) == 20, f"Expected 20 training FOV masks, found {len(train_masks)}"
    assert len(test_images) == 20, f"Expected 20 test images, found {len(test_images)}"
    assert len(test_masks) == 20, f"Expected 20 test FOV masks, found {len(test_masks)}"
    
    print("Dataset verification SUCCESSFUL!")
    print("Strategy:")
    print("  - Split 20 labeled training images into Train (e.g. 16) & Validation (e.g. 4) at IMAGE level.")
    print("  - Validate ground-truth metrics (Accuracy, Sensitivity, Specificity, Dice, AUC) on Validation set.")
    print("  - Generate final held-out predictions for 20 Test set images within FOV mask.")
    print("=" * 60)

if __name__ == "__main__":
    setup_dataset()
