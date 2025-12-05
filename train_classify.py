from ultralytics import YOLO

def main():
    # 1. Load a pre-trained Classification Model
    # 'yolov8n-cls.pt' is smaller and faster than the detection model
    model = YOLO('yolov8n-cls.pt') 

    # 2. Train the model
    results = model.train(
        data='species_dataset', # Path to your folder from Step 1
        epochs=20,              # Classification learns faster than detection
        imgsz=128,              # Tree crops are usually small squares
        batch=16,
        name='species_model',   # Saves to runs/classify/species_model
        device=0                # Use your GPU
    )

    # 3. Validate
    metrics = model.val()
    print(f"Top-1 Accuracy: {metrics.top1}") # How often it guesses the right species

if __name__ == "__main__":
    main()