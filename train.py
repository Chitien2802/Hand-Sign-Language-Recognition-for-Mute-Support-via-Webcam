#Name: Nguyen Chi Tien
#Location: Department of Civil Engineering and Environment, College of Engineering, Myongji University, 116 Myongji-ro, Cheoin-gu, Yongin, Gyeonggy-do 449-728, Korea.

from ultralytics import YOLO
import os
import sys
import shutil
from collections import Counter

def rename_folders(base_path):
    print(f"\n{'=' * 100}")
    print("RENAMING FOLDERS...")
    print("=" * 100 + "\n")

    folders = ['train', 'val', 'test']

    for folder in folders:
        folder_path = os.path.join(base_path, folder)

        if not os.path.exists(folder_path):
            print(f" Folder not found: {folder_path}")
            continue

        
        old_images = os.path.join(folder_path, 'image')
        new_images = os.path.join(folder_path, 'images')

        if os.path.exists(old_images) and not os.path.exists(new_images):
            try:
                os.rename(old_images, new_images)
                print(f" {folder}/image  → {folder}/images")
            except Exception as e:
                print(f" Failed to rename {folder}/image: {e}")
        elif os.path.exists(new_images):
            print(f" {folder}/images  (already exists)")

        # Rename label → labels
        old_labels = os.path.join(folder_path, 'label')
        new_labels = os.path.join(folder_path, 'labels')

        if os.path.exists(old_labels) and not os.path.exists(new_labels):
            try:
                os.rename(old_labels, new_labels)
                print(f" {folder}/label  → {folder}/labels")
            except Exception as e:
                print(f" Failed to rename {folder}/label: {e}")
        elif os.path.exists(new_labels):
            print(f" {folder}/labels  (already exists)")



#  ANALYZE LABEL FILES
def analyze_labels(labels_dir, split_name="train"):
  
    if not os.path.exists(labels_dir):
        print(f"  [{split_name}] Labels folder not found: {labels_dir}")
        return

    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    if not label_files:
        print(f"  [{split_name}] No label files found in: {labels_dir}")
        return

    class_counter = Counter()
    invalid_lines = 0

    for fname in label_files:
        path = os.path.join(labels_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    try:
                        cls_id = int(parts[0])
                        class_counter[cls_id] += 1
                    except Exception:
                        invalid_lines += 1
        except Exception as e:
            print(f"   Error reading {path}: {e}")

    print(f"\n  [{split_name}] Class distribution (box counts):")
    if class_counter:
        for cls_id in sorted(class_counter.keys()):
            print(f"    - Class {cls_id}: {class_counter[cls_id]} boxes")
    else:
        print("    (no valid boxes found)")

    if invalid_lines > 0:
        print(f"    WARNING: {invalid_lines} invalid label lines (cannot parse class id).")

#  MAIN
def main():
    print("\n" + "=" * 100)
    print("TRAIN YOLO11 FOR HAND GESTURE DETECTION - FULL AUTOMATIC")
    print("=" * 100)
  
    print("\n PATHS:\n")
    
    #đường dẫn

    dataset_folder = r''
    yaml_file = r''
    output_folder = r''

    print(f"Dataset:  {dataset_folder}")
    print(f"YAML:     {yaml_file}")
    print(f"Output:   {output_folder}")

  
    rename_folders(dataset_folder)

    # ===== VERIFY PATHS =====
    print(f"\n{'=' * 100}")
    print("VERIFYING PATHS...")
    print("=" * 100 + "\n")

    errors = []

    # Check dataset folder
    if not os.path.exists(dataset_folder):
        errors.append(f"Dataset folder not found: {dataset_folder}")
    else:
        print("Dataset folder exists")

  
    def count_files(img_dir, lbl_dir, split_name):
        img_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        img_count = len([f for f in os.listdir(img_dir) if f.lower().endswith(img_exts)]) if os.path.exists(img_dir) else 0
        lbl_count = len([f for f in os.listdir(lbl_dir) if f.lower().endswith('.txt')]) if os.path.exists(lbl_dir) else 0
        print(f" {split_name}: {img_count} images, {lbl_count} labels")
        if img_count == 0 or lbl_count == 0:
            errors.append(f"{split_name} folder is empty or incomplete")
        if img_count != lbl_count:
            print(f"  WARNING: {split_name} images != labels (check missing or extra labels)")

    # Check train folder
    train_path = os.path.join(dataset_folder, 'train')
    train_images = os.path.join(train_path, 'images')
    train_labels = os.path.join(train_path, 'labels')

    if not os.path.exists(train_path):
        errors.append(f"Train folder not found: {train_path}")
    else:
        count_files(train_images, train_labels, "Train")

    # Check val folder
    val_path = os.path.join(dataset_folder, 'val')
    val_images = os.path.join(val_path, 'images')
    val_labels = os.path.join(val_path, 'labels')

    if not os.path.exists(val_path):
        errors.append(f"Val folder not found: {val_path}")
    else:
        count_files(val_images, val_labels, "Val")

    # Check test folder (optional)
    test_path = os.path.join(dataset_folder, 'test')
    if os.path.exists(test_path):
        test_images = os.path.join(test_path, 'images')
        test_labels = os.path.join(test_path, 'labels')
        count_files(test_images, test_labels, "Test")

    # Check YAML
    if not os.path.exists(yaml_file):
        errors.append(f"YAML file not found: {yaml_file}")
    else:
        print(f" YAML file exists")

    if errors:
        print(f"\n ERRORS:\n")
        for error in errors:
            print(f"  • {error}")
        sys.exit(1)

    print(f"\n All paths verified!")

    # ===== ANALYZE CLASS DISTRIBUTION  =====
    analyze_labels(train_labels, split_name="Train")
    analyze_labels(val_labels, split_name="Val")

    # ===== LOAD MODEL =====
    print(f"\n{'=' * 100}")
    print(" LOADING YOLO11 MODEL...")
    print("=" * 100 + "\n")

    try:
        model = YOLO('yolo11n.pt')   
        print(" YOLO11 Nano loaded successfully\n")
    except Exception as e:
        print(f" Error: {e}")
        sys.exit(1)

    # ===== TRAINING CONFIG =====
    print("=" * 100)
    print("  TRAINING CONFIGURATION")
    print("=" * 100 + "\n")

    config = {
        'Epochs': '300',
        'Batch size': '64',
        'Learning rate (lr0)': '0.001',
        'Final LR (lrf)': '0.0001',
        'Optimizer': 'ADAM',
        'Momentum': '0.937',
        'Weight decay (L2)': '0.0005',
        'Early stopping': 'Patience 30',
        'Dropout': '0.1',
        'Input size': '640x640',
        'Augmentation': 'HSV, degrees=10, translate=0.1, scale=0.5, flipud=0.5, fliplr=0.5'
    }

    for key, value in config.items():
        print(f"  • {key}: {value}")

    print()

    # = TRAINING =====
    print("=" * 100)
    print("STARTING TRAINING...")
    print("=" * 100 + "\n")

    try:
        results = model.train(
            data=yaml_file,
            epochs=300,
            batch=32,
            imgsz=640,
            patience=30,
            device=0,            
            lr0=0.001,           
            lrf=0.0001,
            optimizer='ADAM',
            momentum=0.937,
            weight_decay=0.0005,
            dropout=0.1,         
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10,
            translate=0.1,
            scale=0.5,
            flipud=0.5,
            fliplr=0.5,
            val=True,
            save=True,
            save_period=10,
            project=output_folder,
            name='gesture_detection_v2',
            exist_ok=True,
            verbose=True,
            seed=42,            
        )

        print("\n" + "=" * 100)
        print(" TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 100)

    except KeyboardInterrupt:
        print("\n Training stopped by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"\n Training error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ===== RESULTS =====
    print(f"\n TRAINING RESULTS:\n")

    results_dir = results.save_dir
    best_model_path = os.path.join(results_dir, 'weights', 'best.pt')
    last_model_path = os.path.join(results_dir, 'weights', 'last.pt')
    results_csv = os.path.join(results_dir, 'results.csv')
    results_png = os.path.join(results_dir, 'results.png')

    print(f" Output directory: {results_dir}\n")

    print(f"Model files:")
    print(f"  Best model: {best_model_path}")
    print(f"   Last model: {last_model_path}")

    print(f"\n Metrics and charts:")
    print(f"   CSV metrics: {results_csv}")
    print(f"   PNG graphs: {results_png}")
    print(f"     ├── Loss curve (box_loss, cls_loss, dfl_loss)")
    print(f"     ├── Accuracy curve (mAP50, mAP50-95)")
    print(f"     ├── Precision curve")
    print(f"     └── Recall curve")

    # ===== VALIDATION METRICS =====
    print(f"\n{'=' * 100}")
    print(" VALIDATION METRICS")
    print("=" * 100 + "\n")

    try:
        best = YOLO(best_model_path)
        metrics = best.val()

        map50 = metrics.box.map50 * 100
        map_value = metrics.box.map * 100
        precision = metrics.box.mp * 100
        recall = metrics.box.mr * 100

        print(f"   Accuracy (mAP50):     {map50:.2f}%")
        print(f"   Accuracy (mAP50-95):  {map_value:.2f}%")
        print(f"   Precision:            {precision:.2f}%")
        print(f"   Recall:               {recall:.2f}%")

        print(f"\n FINAL ACCURACY: {map_value:.2f}%")

    except Exception as e:
        print(f"  Could not calculate metrics: {e}")

    
    print(f"\n{'=' * 100}")
    print(" NEXT STEPS")
    print("=" * 100 + "\n")

    print(f"Model training completed!\n")
    print(f" Best model location:")
    print(f"   {best_model_path}\n")
    print(f" Next: Use this best.pt in your webcam inference script.")
    print(f"\n{'=' * 100}\n")


if __name__ == '__main__':
    main()
