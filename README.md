# train-md-yolov5

#Clone yolov5

#Use Train 

train.py --img 640 --batch 16 --epochs 50 --data custom_data.yaml --weights yolov5s.pt

#Use Detect 

python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.3 --source your_image.jpg

