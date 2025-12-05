import os
import glob
from ultralytics import YOLO
from tqdm import tqdm

# Settings
VALID_IMAGES_DIR = "yolo_dataset/valid/images"
VALID_LABELS_DIR = "yolo_dataset/valid/labels"
MODEL_PATH = "runs/detect/yolo_tree_mixed3/weights/best.pt"

def count_lines_in_txt(path):
    if not os.path.exists(path): return 0
    with open(path, 'r') as f:
        return len(f.readlines())

def main():
    # 1. Load the trained YOLO model
    model = YOLO(MODEL_PATH)
    
    image_paths = glob.glob(os.path.join(VALID_IMAGES_DIR, "*.*"))
    print(f"Evaluating YOLO on {len(image_paths)} images...")

    total_mae = 0.0
    total_images = 0
    exact_matches = 0

    # 2. Run inference
    # stream=True is efficient for large datasets
    results = model.predict(source=VALID_IMAGES_DIR, conf=0.25, stream=True, verbose=False)

    for result in tqdm(results, total=len(image_paths)):
        # Predicted count
        pred_count = len(result.boxes)
        
        # Ground Truth count
        # Find matching .txt file
        img_name = os.path.basename(result.path)
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        lbl_path = os.path.join(VALID_LABELS_DIR, txt_name)
        true_count = count_lines_in_txt(lbl_path)

        # Calculate Error
        err = abs(pred_count - true_count)
        total_mae += err
        if err == 0:
            exact_matches += 1
        total_images += 1

    # 3. Print Results
    final_mae = total_mae / total_images
    accuracy = (exact_matches / total_images) * 100
    
    print("\n" + "="*40)
    print(f"YOLO COMPARISON RESULTS")
    print("="*40)
    print(f"Total Images: {total_images}")
    print(f"Mean Absolute Error (MAE): {final_mae:.3f} (Lower is better)")
    print(f"Exact Count Accuracy:      {accuracy:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()