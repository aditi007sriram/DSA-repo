# detect_ambiguous.py
from ultralytics import YOLO
import cv2
import numpy as np
import os

model = YOLO('yolov8n.pt')
empty_dir = 'ambiguous_empty'  # New folder
all_detections = []

print("Detecting hallucinations in ambiguous scenes...\n")

for img_file in sorted(os.listdir(empty_dir)):
    if not img_file.endswith(('.jpg', '.png')):
        continue
    
    img_path = os.path.join(empty_dir, img_file)
    results = model(img_path, conf=0.10, verbose=False)  # Low threshold
    
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    boxes = results[0].boxes
    
    print(f"{img_file}: {len(boxes)} detections")
    
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        cls = int(box.cls[0].cpu().numpy())
        class_name = model.names[cls]
        
        center_x = ((x1 + x2) / 2) / w
        center_y = ((y1 + y2) / 2) / h
        
        all_detections.append({
            'class': class_name,
            'confidence': conf,
            'center_x': center_x,
            'center_y': center_y
        })
        
        print(f"  - {class_name}: {conf:.2f} at ({center_x:.2f}, {center_y:.2f})")
        
        # Draw box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), 
                     (0, 0, 255), 3)
        label = f"{class_name} {conf:.2f}"
        cv2.putText(img, label, (int(x1), int(y1)-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    if len(boxes) > 0:
        output_filename = f'HALLUC_{img_file}'
        cv2.imwrite(output_filename, img)
        print(f"  ✓ Saved: {output_filename}")

print(f"\nTotal hallucinations: {len(all_detections)}")

# Combine with previous detections
try:
    old_detections = np.load('hallucination_detections.npy', allow_pickle=True).tolist()
    all_detections.extend(old_detections)
    print(f"Combined with {len(old_detections)} previous detections")
except:
    pass

np.save('hallucination_detections.npy', all_detections)
print(f"✓ Total saved: {len(all_detections)} hallucinations")