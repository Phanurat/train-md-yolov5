import torch
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

# โหลดโมเดล
model = torch.load('yolov5/runs/train/exp/weights/best.pt', map_location=torch.device('cpu'))['model'].float()  # โหลดโมเดล
model.eval()  # ตั้งค่าโมเดลเป็นโหมดทดสอบ

# อ่านภาพ
img = cv2.imread('test_img/test.png')  # อ่านภาพ
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # แปลงภาพจาก BGR เป็น RGB

# ปรับขนาดภาพให้เหมาะสมกับ YOLOv5
img_resized = cv2.resize(img_rgb, (640, 640))  # ปรับขนาดภาพเป็น 640x640

# แปลงภาพเป็น tensor
img_tensor = torch.from_numpy(img_resized).float()  # เปลี่ยนเป็น tensor
img_tensor = img_tensor.permute(2, 0, 1)  # แปลงเป็นรูปแบบ CxHxW
img_tensor /= 255.0  # สเกลภาพให้ค่าอยู่ระหว่าง 0-1
img_tensor = img_tensor.unsqueeze(0)  # เพิ่มมิติ batch

# ทำการทำนาย
with torch.no_grad():  # ไม่ต้องคำนวณ gradients ในระหว่างการทำนาย
    results = model(img_tensor)  # ทำนายบนภาพ

# แสดงผลลัพธ์
results.show()  # แสดงผล

# หรือบันทึกผล
results.save()  # บันทึกภาพที่มี bounding box
