from ultralytics import YOLO

def main():
    # 1. Load the pre-trained Classification Model
    # 'yolov8n-cls.pt' is designed for whole-image classification
    model = YOLO('yolov8n-cls.pt') 

    # 2. Train it
    results = model.train(
        data='species_dataset', # Point to the folder you created in Step 1
        epochs=30,              # Classification learns very fast (20 is usually enough)
        imgsz=224,              # Standard size for classification models
        batch=16,
        name='my_species_model',
        device=0
    )

    # 3. Validate to see accuracy
    metrics = model.val()
    print(f"Top-1 Accuracy: {metrics.top1:.2f}") # E.g., 0.95 means 95% correct

if __name__ == "__main__":
    main()