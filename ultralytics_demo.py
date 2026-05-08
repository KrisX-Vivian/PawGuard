"""
PawGuard Demo with Ultralytics YOLO
This demo uses YOLOv8 to detect cats and dogs in images.

Usage:
  # Single image detection
  python src/ultralytics_demo.py [image_path]
  
  # Batch detection from datasets
  python src/ultralytics_demo.py --batch --dataset1 data/dataset/Oxford-IIIT-Pet-dataset/images --dataset2 data/dataset/kaggle-cat-vs-dog-dataset --count 10
"""

import sys
import io
import time
import argparse
import random
# Set UTF-8 encoding to fix Chinese character display issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from ultralytics import YOLO
import cv2
import os

def detect_single_image(model, image_path, output_dir):
    """Detect cats/dogs in a single image and save result"""
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return None

    print(f"[INFO] Processing: {os.path.basename(image_path)}")
    
    # Detect - only cat and dog (COCO classes 15 and 16)
    results = model(image_path, classes=[15, 16])  # 15=cat, 16=dog

    # Process results
    for i, result in enumerate(results):
        # Generate annotated image
        annotated_frame = result.plot()
        
        # Save annotated image with timestamp to avoid overwriting
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{name_without_ext}_{timestamp}.jpg")
        cv2.imwrite(output_path, annotated_frame)
        
        # Collect detection results
        detections = []
        if len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                confidence = float(box.conf[0])
                detections.append({"class": cls_name, "confidence": confidence})
                print(f"  - {cls_name}: {confidence:.2%}")
        else:
            print("  - No cats or dogs detected")
        
        return {"image": filename, "detections": detections, "output": output_path}

def batch_detect_from_datasets(model, dataset1_path, dataset2_path, count=10):
    """Batch detect from two datasets and save results separately"""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create output directories for each dataset
    output_dir1 = os.path.join(BASE_DIR, "demo", "Oxford-IIIT-Pet-dataset")
    output_dir2 = os.path.join(BASE_DIR, "demo", "kaggle-cat-vs-dog-dataset")
    os.makedirs(output_dir1, exist_ok=True)
    os.makedirs(output_dir2, exist_ok=True)
    
    # Get list of image files from dataset 1 (Oxford-IIIT-Pet-dataset)
    print(f"\n{'='*60}")
    print(f"Processing Dataset 1: Oxford-IIIT-Pet-dataset")
    print(f"{'='*60}")
    
    dataset1_images = []
    if os.path.exists(dataset1_path):
        for file in os.listdir(dataset1_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                dataset1_images.append(os.path.join(dataset1_path, file))
    
    if len(dataset1_images) == 0:
        print(f"[ERROR] No images found in {dataset1_path}")
    else:
        # Randomly select 'count' images
        selected1 = random.sample(dataset1_images, min(count, len(dataset1_images)))
        print(f"Selected {len(selected1)} images from {len(dataset1_images)} total")
        
        # Process each image
        results1 = []
        for img_path in selected1:
            result = detect_single_image(model, img_path, output_dir1)
            if result:
                results1.append(result)
        
        # Summary for dataset 1
        print(f"\n[SUMMARY] Dataset 1 (Oxford-IIIT-Pet-dataset):")
        print(f"  Total images processed: {len(results1)}")
        cats_detected = sum(1 for r in results1 for d in r['detections'] if d['class'] == 'cat')
        dogs_detected = sum(1 for r in results1 for d in r['detections'] if d['class'] == 'dog')
        print(f"  Cats detected: {cats_detected}")
        print(f"  Dogs detected: {dogs_detected}")
        print(f"  Results saved to: {output_dir1}")
    
    # Get list of image files from dataset 2 (kaggle-cat-vs-dog-dataset)
    print(f"\n{'='*60}")
    print(f"Processing Dataset 2: Kaggle Cat vs Dog Dataset")
    print(f"{'='*60}")
    
    dataset2_images = []
    if os.path.exists(dataset2_path):
        # Kaggle dataset has nested structure: test_set/test_set/cats/ and test_set/test_set/dogs/
        for root, dirs, files in os.walk(dataset2_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    dataset2_images.append(os.path.join(root, file))
    
    if len(dataset2_images) == 0:
        print(f"[ERROR] No images found in {dataset2_path}")
    else:
        # Randomly select 'count' images
        selected2 = random.sample(dataset2_images, min(count, len(dataset2_images)))
        print(f"Selected {len(selected2)} images from {len(dataset2_images)} total")
        
        # Process each image
        results2 = []
        for img_path in selected2:
            result = detect_single_image(model, img_path, output_dir2)
            if result:
                results2.append(result)
        
        # Summary for dataset 2
        print(f"\n[SUMMARY] Dataset 2 (Kaggle Cat vs Dog):")
        print(f"  Total images processed: {len(results2)}")
        cats_detected = sum(1 for r in results2 for d in r['detections'] if d['class'] == 'cat')
        dogs_detected = sum(1 for r in results2 for d in r['detections'] if d['class'] == 'dog')
        print(f"  Cats detected: {cats_detected}")
        print(f"  Dogs detected: {dogs_detected}")
        print(f"  Results saved to: {output_dir2}")
    
    # Overall summary
    print(f"\n{'='*60}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset 1 (Oxford-IIIT-Pet): {len(results1) if 'results1' in dir() else 0} images processed")
    print(f"Dataset 2 (Kaggle): {len(results2) if 'results2' in dir() else 0} images processed")
    print(f"Results saved to separate directories under demo/")

def main():
    print("=" * 60)
    print("PawGuard Demo with Ultralytics YOLO")
    print("=" * 60)

    # Parse arguments
    parser = argparse.ArgumentParser(description='PawGuard Cat/Dog Detection Demo')
    parser.add_argument('image', nargs='?', help='Path to image file (relative to project root)')
    parser.add_argument('--batch', action='store_true', help='Enable batch mode for dataset processing')
    parser.add_argument('--dataset1', default='data/dataset/Oxford-IIIT-Pet-dataset/images', 
                        help='Path to first dataset (Oxford-IIIT-Pet)')
    parser.add_argument('--dataset2', default='data/dataset/kaggle-cat-vs-dog-dataset', 
                        help='Path to second dataset (Kaggle)')
    parser.add_argument('--count', type=int, default=10, help='Number of images to select from each dataset')
    args = parser.parse_args()

    # Load pre-trained YOLOv8n model (nano version, fast)
    print("[WAIT] Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")  # Auto-downloaded to ~/.cache/ultralytics/
    print("[OK] Model loaded successfully!")

    if args.batch:
        # Batch mode: process datasets
        batch_detect_from_datasets(model, args.dataset1, args.dataset2, args.count)
    else:
        # Single image mode
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Default image path if not provided
        if args.image is None:
            image_path = os.path.join(BASE_DIR, "data", "test.jpg")
        elif not os.path.isabs(args.image):
            image_path = os.path.join(BASE_DIR, args.image)
        else:
            image_path = args.image
        
        output_dir = os.path.join(BASE_DIR, "demo")
        os.makedirs(output_dir, exist_ok=True)
        
        detect_single_image(model, image_path, output_dir)
        
        print("\n" + "=" * 60)
        print("[SUCCESS] Demo completed!")

if __name__ == "__main__":
    main()
