from ultralytics import YOLO

def main():
    # Load a model
    # 'yolov8n.pt' is the "nano" version (fastest). 
    # Use 'yolov8s.pt' (small) or 'yolov8m.pt' (medium) for better accuracy.
    model = YOLO("yolov8n.pt") 

    # Train the model
    results = model.train(
        data="tree_config.yaml",  
        epochs=50,                # Good choice for mixed data
        imgsz=640,                
        batch=16,                 # UPDATED: Use 16 or 32 since you have 16GB VRAM
        name="yolo_tree_mixed",   # UPDATED: Changed name to avoid confusion with the old coconut model
        device=0                  # UPDATED: Must be 0 for your 4060 Ti
    )

    # Validate the model
    metrics = model.val()
    print(f"mAP@50: {metrics.box.map50}")
    print(f"mAP@50-95: {metrics.box.map}")

    # Export to ONNX (useful for deployment later)
    model.export(format="onnx")

if __name__ == "__main__":
    main()