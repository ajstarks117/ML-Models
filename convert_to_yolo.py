import os
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image

# Define your paths
TRAIN_DIR = "train"  # Your current train folder
VALID_DIR = "valid"  # Your current valid folder
OUTPUT_DIR = "yolo_dataset"  # New folder for YOLO data

def convert_box(size, box):
    """Converts min/max coordinates to normalized center/width/height"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def process_folder(folder_path, subset_name):
    # Create YOLO directory structure
    img_save_path = os.path.join(OUTPUT_DIR, subset_name, "images")
    lbl_save_path = os.path.join(OUTPUT_DIR, subset_name, "labels")
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(lbl_save_path, exist_ok=True)

    files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]
    
    for xml_file in tqdm(files, desc=f"Converting {subset_name}"):
        xml_path = os.path.join(folder_path, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Image info
        img_filename = root.find('filename').text
        # Safety check if filename in XML doesn't match file on disk
        if not os.path.exists(os.path.join(folder_path, img_filename)):
             # try replacing extension or using the xml stem
             stem = os.path.splitext(xml_file)[0]
             possible_exts = ['.jpg', '.jpeg', '.png', '.webp']
             for ext in possible_exts:
                 if os.path.exists(os.path.join(folder_path, stem + ext)):
                     img_filename = stem + ext
                     break
        
        # Get image size from XML or load image to check
        size = root.find('size')
        if size is not None:
            w = int(size.find('width').text)
            h = int(size.find('height').text)
        else:
            # Fallback: load image
            with Image.open(os.path.join(folder_path, img_filename)) as img:
                w, h = img.size

        # Convert boxes
        yolo_lines = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name != "trees": continue  # Skip if not tree
            
            bndbox = obj.find('bndbox')
            b = (float(bndbox.find('xmin').text), float(bndbox.find('xmax').text),
                 float(bndbox.find('ymin').text), float(bndbox.find('ymax').text))
            bb = convert_box((w, h), b)
            
            # class_id is 0 for 'trees'
            yolo_lines.append(f"0 {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}")

        # Save Label File
        txt_filename = os.path.splitext(img_filename)[0] + ".txt"
        with open(os.path.join(lbl_save_path, txt_filename), 'w') as f:
            f.write('\n'.join(yolo_lines))
            
        # Copy Image File
        src_img = os.path.join(folder_path, img_filename)
        dst_img = os.path.join(img_save_path, img_filename)
        shutil.copy(src_img, dst_img)

if __name__ == "__main__":
    process_folder(TRAIN_DIR, "train")
    process_folder(VALID_DIR, "valid")
    print(f"Conversion complete! Data saved to {OUTPUT_DIR}")