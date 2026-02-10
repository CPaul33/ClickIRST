"""
IRST dataset sample weight calculation script
Calculate the difficulty score for each sample and generate a weight file
"""
import os
import cv2
import numpy as np
import pickle
from pathlib import Path
from scipy import stats
from skimage.measure import regionprops
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

def compute_target_difficulty(image, mask):
    """Calculate difficulty score for a single target"""
    # 1. Target size (smaller is harder) - use logarithmic scale to amplify difficulty of small targets
    target_area = np.sum(mask)
    if target_area == 0:
        size_score = 1.0
    else:
        # Use logarithmic transformation to make small target difficulty more prominent
        size_score = min(1.0, 2.0 / np.log(target_area + 1))

    # 2. Target contrast (lower contrast is harder) - improved contrast calculation
    target_pixels = image[mask > 0]

    # Calculate local background (area around the target)
    kernel_size = 5
    dilated_mask = cv2.dilate(mask.astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8))
    local_bg_mask = dilated_mask - mask
    local_bg_pixels = image[local_bg_mask > 0]

    if len(target_pixels) > 0 and len(local_bg_pixels) > 0:
        target_mean = np.mean(target_pixels)
        bg_mean = np.mean(local_bg_pixels)
        contrast_diff = abs(target_mean - bg_mean)
        # Use exponential decay function, low contrast targets get higher scores
        contrast_score = np.exp(-contrast_diff / 30.0)  # 30 is a tuning parameter
    else:
        contrast_score = 0.8  # Default medium difficulty

    # 3. Target shape complexity (more irregular is harder)
    props = regionprops(mask.astype(int))
    if props:
        eccentricity = props[0].eccentricity
        solidity = props[0].solidity
        # Shape complexity calculation, irregular shapes get higher scores
        shape_score = (1 - solidity) * (1 + eccentricity * 0.5)
        shape_score = min(1.0, shape_score)  # Appropriately amplify
    else:
        shape_score = 0.5  # Default medium difficulty

    # Comprehensive difficulty score - adjust weight allocation
    difficulty = size_score * 0.3 + contrast_score * 0.4 + shape_score * 0.3
    return min(1.0, difficulty)  # Appropriately amplify overall

def compute_image_complexity(image):
    """Calculate image background complexity - evaluate using multiple features"""
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image

    # Normalize to 0-255
    gray_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    complexities = []

    # 1. GLCM features
    glcm = graycomatrix(gray_image, [1], [0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    # Normalize these features
    glcm_complexity = (min(contrast/10, 1.0) + (1 - min(abs(correlation), 1.0)) +
                      (1 - min(energy, 1.0)) + (1 - min(homogeneity, 1.0))) / 4
    complexities.append(glcm_complexity)

    # 2. Image gradient features (edge complexity)
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    edge_complexity = np.mean(gradient_magnitude) / 50.0  # Normalize
    complexities.append(min(edge_complexity, 1.0))

    # 3. Image entropy (information complexity)
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    entropy_complexity = entropy / 8.0  # Normalize (max entropy approx 8)
    complexities.append(min(entropy_complexity, 1.0))

    # Comprehensive complexity
    final_complexity = np.mean(complexities)

    return min(1.0, final_complexity)

def compute_irst_weights(dataset_path, split='train', output_path=None, a=1.2, b=0.8):
    """Calculate IRST dataset sample weights"""
    dataset_path = Path(dataset_path)
    images_path = dataset_path / 'images'
    masks_path = dataset_path / 'masks_inst'

    # Read sample list
    with open(dataset_path / 'img_idx' / f'{split}.txt', 'r') as f:
        samples = [x.strip() for x in f.readlines()]

    weights = []

    print(f"Calculating weights for {len(samples)} samples...")
    for idx, sample_name in enumerate(tqdm(samples)):
        # Read image and mask
        image_path = images_path / f'{sample_name}.png'
        mask_path = masks_path / f'{sample_name}.png'

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Warning: Cannot read sample {sample_name}")
            continue

        # Calculate image background complexity
        bg_complexity = compute_image_complexity(image)

        # Calculate difficulty for each target
        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids > 0]  # Remove background

        if len(instance_ids) == 0:
            # Case with no targets
            target_difficulty = 1.0
        else:
            target_difficulties = []
            for obj_id in instance_ids:
                obj_mask = (mask == obj_id).astype(np.uint8)
                difficulty = compute_target_difficulty(image, obj_mask)
                target_difficulties.append(difficulty)

            # Use maximum difficulty to represent image difficulty
            target_difficulty = max(target_difficulties)

        # Comprehensive score - use more balanced combination
        # For IRST small targets, both background complexity and target difficulty are important, but avoid bias towards hard samples
        final_score = target_difficulty * (a + bg_complexity * b)

        weights.append((idx, sample_name, final_score))

    # Improved normalization method - use percentile normalization
    scores = np.array([x[2] for x in weights])

    # Calculate percentiles
    p10 = np.percentile(scores, 10)
    p90 = np.percentile(scores, 90)

    # Use percentiles for normalization to avoid extreme values influence
    normalized_scores = (scores - p10) / (p90 - p10 + 1e-8)

    # Clip to [0, 1] range
    normalized_scores = np.clip(normalized_scores, 0, 1)

    # Use non-linear transformation to focus more on medium difficulty samples
    # Square root transformation: relatively increase medium scores, relatively decrease very high scores
    normalized_scores = np.sqrt(normalized_scores)

    # Update weights
    for i in range(len(weights)):
        weights[i] = (weights[i][0], weights[i][1], normalized_scores[i])

    # Save weight file
    if output_path is None:
        output_path = dataset_path / f'irst_{split}_weights.pkl'

    with open(output_path, 'wb') as f:
        pickle.dump(weights, f)

    print(f"Weights file saved to: {output_path}")
    print(f"Weights statistics: min={normalized_scores.min():.3f}, max={normalized_scores.max():.3f}, mean={normalized_scores.mean():.3f}")

    return weights

def analyze_weights_distribution(weights):
    """Analyze weights distribution"""
    scores = np.array([x[2] for x in weights])

    print("\nWeights Distribution Analysis:")
    print(f"Total samples: {len(scores)}")
    print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"Mean score: {scores.mean():.3f}")
    print(f"Score std dev: {scores.std():.3f}")

    # Quantile analysis
    quantiles = np.quantile(scores, [0.1, 0.25, 0.5, 0.75, 0.9])
    print(f"10% quantile: {quantiles[0]:.3f}")
    print(f"25% quantile: {quantiles[1]:.3f}")
    print(f"50% quantile: {quantiles[2]:.3f}")
    print(f"75% quantile: {quantiles[3]:.3f}")
    print(f"90% quantile: {quantiles[4]:.3f}")

    # Difficulty level distribution
    easy_samples = [x[1] for x in weights if x[2] < 0.4]
    medium_samples = [x[1] for x in weights if 0.4 <= x[2] < 0.8]
    hard_samples = [x[1] for x in weights if x[2] >= 0.8]

    easy = len(easy_samples)
    medium = len(medium_samples)
    hard = len(hard_samples)

    print(f"\nDifficulty Distribution:")
    print(f"Easy samples (score<0.4): {easy} ({easy/len(scores)*100:.1f}%)")
    print(f"Medium samples (0.4≤score<0.8): {medium} ({medium/len(scores)*100:.1f}%)")
    print(f"Hard samples (score≥0.8): {hard} ({hard/len(scores)*100:.1f}%)")

    # Output sample names for each difficulty level
    print(f"\nEasy sample names ({easy} total): {easy_samples}")
    print(f"Medium sample names ({medium} total): {medium_samples}")
    print(f"Hard sample names ({hard} total): {hard_samples}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Calculate IRST dataset sample weights')
    parser.add_argument('--dataset_path', type=str, required=True, help='IRST dataset path')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'], help='Dataset split')
    parser.add_argument('--output_path', type=str, help='Output weight file path')

    args = parser.parse_args()

    weights = compute_irst_weights(args.dataset_path, args.split, args.output_path)
    analyze_weights_distribution(weights)
