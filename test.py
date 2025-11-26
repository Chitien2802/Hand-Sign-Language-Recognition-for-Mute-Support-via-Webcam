#Name: Nguyen Chi Tien
#Location: Department of Civil Engineering and Environment, College of Engineering, Myongji University, 116 Myongji-ro, Cheoin-gu, Yongin, Gyeonggy-do 449-728, Korea.

from ultralytics import YOLO
import os
import sys

def main():
    print("\n" + "=" * 100)
    print("TEST ")
    print("=" * 100)
    print("\n PATHS:\n")

    dataset_folder = r''
    yaml_file      = r''
    train_output   = r''

    test_output    = r''
   
    best_model_path = os.path.join(
        train_output,
        'gesture_detection_v2',
        'weights',
        'best.pt'
    )

    print(f"Dataset folder : {dataset_folder}")
    print(f"YAML file      : {yaml_file}")
    print(f"Train output   : {train_output}")
    print(f"Best model     : {best_model_path}")
    print(f"Test output    : {test_output}")

    # ===== VERIFY PATHS =====
    print("\n" + "=" * 100)
    print("VERIFYING PATHS...")
    print("=" * 100 + "\n")

    errors = []

    # 1. check test/images + test/labels
    test_path   = os.path.join(dataset_folder, 'test')
    test_images = os.path.join(test_path, 'images')
    test_labels = os.path.join(test_path, 'labels')

    if not os.path.exists(test_path):
        errors.append(f"Test folder not found: {test_path}")
    else:
        img_count = len([
            f for f in os.listdir(test_images)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]) if os.path.exists(test_images) else 0

        lbl_count = len([
            f for f in os.listdir(test_labels)
            if f.lower().endswith('.txt')
        ]) if os.path.exists(test_labels) else 0

        print(f" Test: {img_count} images, {lbl_count} labels")
        if img_count == 0 or lbl_count == 0:
            errors.append("Test folder is empty or labels/images missing.")

    # 2. YAML
    if not os.path.exists(yaml_file):
        errors.append(f"YAML file not found: {yaml_file}")
    else:
        print(" YAML file exists")

    # 3. best.pt
    if not os.path.exists(best_model_path):
        errors.append(f"Best model not found: {best_model_path}")
    else:
        print(" Best model exists")

    if errors:
        print("\n ERRORS:\n")
        for e in errors:
            print("  •", e)
        print("\nPlease fix the errors above and run again.")
        sys.exit(1)

    print("\n All paths verified! ✅")

    # ===== LOAD BEST MODEL =====
    print("\n" + "=" * 100)
    print("LOADING BEST MODEL...")
    print("=" * 100 + "\n")

    try:
        model = YOLO(best_model_path)
        print(" Best model loaded successfully\n")
    except Exception as e:
        print(f" Error loading model: {e}")
        sys.exit(1)

    # ===== EVALUATION=====
    print("=" * 100)
    print("RUNNING EVALUATION ON TEST SET...")
    print("=" * 100 + "\n")

    try:
        # split='test' → dùng tập test trong data.yaml
        metrics = model.val(
            data=yaml_file,
            split='test',          # dùng tập test (cần khai báo 'test:' trong data.yaml)
            imgsz=640,
            batch=32,
            device=0,            
            save=True,             
            save_txt=True,        
            save_conf=True,        
            save_json=True,        
            project=test_output,   
            name='gesture_test_v1',
            exist_ok=True,
            verbose=True,
        )
        # ===== PRINT METRICS =====
        print("\n" + "=" * 100)
        print("TEST METRICS")
        print("=" * 100 + "\n")

        # metrics.box. là object của Ultralytics
        map50   = metrics.box.map50 * 100      
        map_all = metrics.box.map * 100        
        prec    = metrics.box.mp * 100         
        rec     = metrics.box.mr * 100         

        print(f" mAP@0.5      : {map50:.2f}%")
        print(f" mAP@0.5:0.95 : {map_all:.2f}%")
        print(f" Precision    : {prec:.2f}%")
        print(f" Recall       : {rec:.2f}%")

        print("\nDetailed results saved to:")
        print(f"  {metrics.save_dir}")  

        print("\nDONE TESTING ")

    except Exception as e:
        print(f"\n Test error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 100 + "\n")

if __name__ == "__main__":
    main()
