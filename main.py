import os
import json

# 4. ระบุ path ของไฟล์ JSON ที่อยู่ใน ZIP (แก้ไขตามไฟล์ที่แตกออกมา)
json_file_path = os.path.join('model/train/_annotations.coco.json')

# 5. โหลดไฟล์ JSON
with open(json_file_path, 'r') as f:
    annotations = json.load(f)

# 6. แสดงข้อมูล "info" (ข้อมูลทั่วไปเกี่ยวกับโปรเจกต์)
print("Project Info:")
print(json.dumps(annotations['info'], indent=4))

# 7. แสดงข้อมูล "categories" (หมวดหมู่ของวัตถุที่โมเดลสามารถตรวจจับได้)
print("\nCategories:")
for category in annotations['categories']:
    print(f"ID: {category['id']}, Name: {category['name']}, Supercategory: {category['supercategory']}")

# 8. แสดงข้อมูล "annotations" (แอนโนเทชัน เช่น ข้อมูล bounding boxes ของวัตถุ)
print("\nAnnotations:")
for annotation in annotations['annotations']:
    image_id = annotation['image_id']
    category_id = annotation['category_id']
    bbox = annotation['bbox']  # (x, y, width, height)
    print(f"Image ID: {image_id}, Category ID: {category_id}, Bounding Box: {bbox}")