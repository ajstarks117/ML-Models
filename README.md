# Tree Counting with Faster R-CNN (PyTorch)

This project trains a Faster R-CNN model to detect and count coconut trees from a VOC-style dataset exported by Roboflow. Folders expected: `train/`, `valid/`, `test/` each with paired `.jpg` and `.xml` files.

## Environment

Install dependencies (Windows PowerShell):

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Install CUDA-enabled PyTorch matching your GPU/driver if needed. See `https://pytorch.org/get-started/locally/`.

## Train

By default the scripts validate GPU index and fall back to 0.

```powershell
python train.py --train_dir train --valid_dir valid --epochs 20 --batch_size 4 --gpu 0 --output outputs
```

During validation, the script prints MAE and exact counting Accuracy (%). Checkpoints saved per epoch in `outputs/`, with convenience files `best_mae.pth`, `best_acc.pth`, and a final `final.pth` after training.

To resume:

```powershell
python train.py --train_dir train --valid_dir valid --resume outputs\best_acc.pth --gpu 0
```

## Inference (CLI)

Count trees in a single image or a directory:

```powershell
# Single image
python infer.py --weights outputs\best_acc.pth --input test\some_image.jpg --gpu 0 --score_thresh 0.5

# Directory
python infer.py --weights outputs\best_acc.pth --input test --gpu 0 --score_thresh 0.5
```

## Temporary Flask UI

Start a simple web UI to upload an image and visualize detections and count:

```powershell
$env:WEIGHTS = "outputs\best_acc.pth"   # or outputs\final.pth
$env:GPU = "0"
$env:THRESH = "0.5"
python app.py
```

Then open `http://127.0.0.1:5000/` in your browser, upload an image, and see the predicted boxes and count.

## Notes

- Class name is fixed to `trees` from the annotations. If your labels differ, update `CLASS_NAME` in `voc_dataset.py`.
- Model: `torchvision` Faster R-CNN ResNet50 FPN v2, fine-tuned for 1 class (+background).
- Validation metrics: MAE (lower is better) and exact counting Accuracy (higher is better) at the chosen score threshold.
