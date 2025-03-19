import json

# โหลดข้อมูลจากไฟล์ COCO JSON
with open('C:/Users/USER/Desktop/test-hugging/main/model/train/_annotations.coco.json') as f:
    coco_data = json.load(f)

# ดูโครงสร้างของไฟล์
print(coco_data.keys())

# ดูข้อมูล categories (classes)
categories = coco_data['categories']
print(categories)

import os
import json

# โหลดข้อมูลจากไฟล์ COCO JSON
with open('C:/Users/USER/Desktop/test-hugging/main/model/train/_annotations.coco.json') as f:
    coco_data = json.load(f)

# โฟลเดอร์ที่เก็บภาพ
image_dir = 'C:/Users/USER/Desktop/test-hugging/main/model/train/images'

# โฟลเดอร์ที่จะเก็บไฟล์ label (YOLO format)
output_dir = 'C:/Users/USER/Desktop/test-hugging/main/model/train/labels'
os.makedirs(output_dir, exist_ok=True)

# แปลงข้อมูล annotation ให้เป็น YOLO format
for annotation in coco_data['annotations']:
    image_id = annotation['image_id']
    bbox = annotation['bbox']  # [x, y, width, height]
    class_id = annotation['category_id']

    # คำนวณค่าที่ normalize ของ Bounding Box
    image_width = coco_data['images'][image_id]['width']
    image_height = coco_data['images'][image_id]['height']
    
    x_center = (bbox[0] + bbox[2] / 2) / image_width
    y_center = (bbox[1] + bbox[3] / 2) / image_height
    width = bbox[2] / image_width
    height = bbox[3] / image_height

    # เขียนข้อมูล label ลงในไฟล์ .txt
    label_file = os.path.join(output_dir, f'{image_id}.txt')
    with open(label_file, 'a') as f:
        f.write(f'{class_id} {x_center} {y_center} {width} {height}\n')
