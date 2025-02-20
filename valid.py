import os
import numpy as np
import cv2
from sklearn.metrics import mean_absolute_error

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-6)

def iou_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + 1e-6)

def calculate_metrics(gt_folder, pred_folder):
    subfolders = os.listdir(gt_folder)

    total_mae = 0
    total_dice = 0
    total_iou = 0
    image_count = 0

    for subfolder in subfolders:
        gt_subfolder = os.path.join(gt_folder, subfolder)
        pred_subfolder = os.path.join(pred_folder, subfolder)

        if not os.path.isdir(gt_subfolder) or not os.path.isdir(pred_subfolder):
            continue

        gt_images = sorted(os.listdir(gt_subfolder))
        pred_images = sorted(os.listdir(pred_subfolder))

        for gt_image_name, pred_image_name in zip(gt_images, pred_images):
            gt_image_path = os.path.join(gt_subfolder, gt_image_name)
            pred_image_path = os.path.join(pred_subfolder, pred_image_name)

            gt_image = cv2.imread(gt_image_path, cv2.IMREAD_GRAYSCALE)
            pred_image = cv2.imread(pred_image_path, cv2.IMREAD_GRAYSCALE)

            if gt_image is None or pred_image is None:
                continue

            # Resize GT image to 224x224
            gt_image = cv2.resize(gt_image, (224, 224), interpolation=cv2.INTER_LINEAR)
            pred_image = cv2.resize(pred_image, (224, 224), interpolation=cv2.INTER_LINEAR)

            # Normalize images to binary (0 and 1)
            gt_image = (gt_image > 127).astype(np.uint8)
            pred_image = (pred_image > 127).astype(np.uint8)

            mae = mean_absolute_error(gt_image.flatten(), pred_image.flatten())
            dice = dice_coefficient(gt_image, pred_image)
            iou = iou_score(gt_image, pred_image)

            total_mae += mae
            total_dice += dice
            total_iou += iou
            image_count += 1

    avg_mae = total_mae / image_count if image_count > 0 else 0
    avg_dice = total_dice / image_count if image_count > 0 else 0
    avg_iou = total_iou / image_count if image_count > 0 else 0

    return {
        'Average MAE': avg_mae,
        'Average Dice': avg_dice,
        'Average IoU': avg_iou,
        'Image Count': image_count
    }

# Replace these with your actual folder paths
gt_folder = "/remote-home/share/24-zhouling/datasets/CAMUS/TestDataset/GT"
# pred_folder = "/remote-home/share/24-zhouling/LSSNet/result/CAMUS"
pred_folder = "/remote-home/lingzhou/Code/ASTR-main/results/lv/log_2025-01-01_20:31:54/figures"

metrics = calculate_metrics(gt_folder, pred_folder)

# Print results
print(f"Average MAE: {metrics['Average MAE']:.4f}")
print(f"Average Dice: {metrics['Average Dice']:.4f}")
print(f"Average IoU: {metrics['Average IoU']:.4f}")
print(f"Total Images Processed: {metrics['Image Count']}")
