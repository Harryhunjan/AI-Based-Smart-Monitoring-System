# cd "c:\Users\hargu\Desktop\Project-OBJECT DETECTION"
# python face_preprocessing.py
"""
Face Recognition Dataset Preprocessing
For Research Paper: AI-based Monitoring System
Author: Your Name
Date: March 2026

This script preprocesses the LFW-DeepFunneled dataset for face recognition tasks.
Preprocessing includes: face detection, resizing, normalization, and data validation.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Setup logging for research documentation
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================
# Set your dataset path here (UPDATE IF NEEDED)
INPUT_PATH = r"C:\Users\hargu\Downloads\Compressed\archive\lfw-deepfunneled"
OUTPUT_PATH = "dataset/processed_faces"
IMG_SIZE = 224
MIN_IMAGES_PER_PERSON = 1  # Minimum images to keep a person in dataset

# ============================================
# STATISTICS TRACKING
# ============================================
stats = {
    'total_people': 0,
    'total_images_processed': 0,
    'total_images_failed': 0,
    'people_with_min_images': 0,
}

def verify_dataset_exists():
    """Verify input dataset is accessible"""
    if not os.path.exists(INPUT_PATH):
        logger.error(f"Dataset not found at: {INPUT_PATH}")
        logger.error("Make sure LFW dataset is extracted at the specified location")
        return False
    
    logger.info(f"✓ Dataset found at: {INPUT_PATH}")
    return True

def create_output_directory():
    """Create output directory structure"""
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        logger.info(f"✓ Created output directory: {OUTPUT_PATH}")
    return OUTPUT_PATH

def preprocess_face_image(image_path):
    """
    Preprocess a single face image
    
    Steps:
    1. Read image
    2. Convert BGR to RGB
    3. Resize to 224x224
    4. Normalize pixel values [0, 1]
    
    Args:
        image_path: Path to input image
        
    Returns:
        Preprocessed image array or None if failed
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Failed to read: {image_path}")
            return None
        
        # Resize to standard size
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Convert BGR to RGB (OpenCV reads as BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img / 255.0
        
        return img
    
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return None

def save_preprocessed_image(image_array, output_path):
    """Save preprocessed image"""
    try:
        # Convert back to uint8 for saving
        img_uint8 = (image_array * 255).astype(np.uint8)
        
        # Convert RGB back to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(output_path, img_bgr)
        return True
    except Exception as e:
        logger.error(f"Failed to save {output_path}: {str(e)}")
        return False

def preprocess_dataset():
    """Main preprocessing pipeline"""
    logger.info("="*60)
    logger.info("STARTING FACE DATASET PREPROCESSING")
    logger.info("="*60)
    
    # Verify dataset
    if not verify_dataset_exists():
        return False
    
    # Create output directory
    create_output_directory()
    
    # Process each person folder
    people_list = sorted([p for p in os.listdir(INPUT_PATH) 
                         if os.path.isdir(os.path.join(INPUT_PATH, p))])
    
    logger.info(f"Found {len(people_list)} people in dataset")
    stats['total_people'] = len(people_list)
    
    for person_idx, person_name in enumerate(people_list, 1):
        person_input_path = os.path.join(INPUT_PATH, person_name)
        person_output_path = os.path.join(OUTPUT_PATH, person_name)
        
        # Create person folder
        if not os.path.exists(person_output_path):
            os.makedirs(person_output_path)
        
        # Process all images for this person
        images = [f for f in os.listdir(person_input_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        
        logger.info(f"[{person_idx}/{len(people_list)}] Processing {person_name} ({len(images)} images)")
        
        processed_count = 0
        for img_name in images:
            img_path = os.path.join(person_input_path, img_name)
            output_img_path = os.path.join(person_output_path, img_name)
            
            # Preprocess image
            preprocessed_img = preprocess_face_image(img_path)
            
            if preprocessed_img is not None:
                # Save preprocessed image
                if save_preprocessed_image(preprocessed_img, output_img_path):
                    processed_count += 1
                    stats['total_images_processed'] += 1
                else:
                    stats['total_images_failed'] += 1
            else:
                stats['total_images_failed'] += 1
        
        # Check minimum images requirement
        if processed_count >= MIN_IMAGES_PER_PERSON:
            stats['people_with_min_images'] += 1
        
        logger.info(f"  ✓ {processed_count}/{len(images)} images processed for {person_name}")
    
    return True

def print_summary():
    """Print and log preprocessing summary"""
    logger.info("="*60)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total People: {stats['total_people']}")
    logger.info(f"Total Images Processed: {stats['total_images_processed']}")
    logger.info(f"Total Images Failed: {stats['total_images_failed']}")
    logger.info(f"People with Sufficient Images: {stats['people_with_min_images']}")
    logger.info(f"Output Directory: {OUTPUT_PATH}")
    logger.info(f"Log File: {log_file}")
    logger.info("="*60)
    logger.info("✓ Preprocessing Complete!")
    logger.info("="*60)

if __name__ == "__main__":
    success = preprocess_dataset()
    print_summary()
    
    if success:
        print("\n✓ Dataset ready for training!")
        print(f"✓ Processed data location: {os.path.abspath(OUTPUT_PATH)}")
    else:
        print("\n✗ Preprocessing failed! Check logs for details.")
