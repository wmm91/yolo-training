from ultralytics import YOLO

model = YOLO('yolov8sbest.pt')

model.export(
    format='coreml', 
    imgsz=768,       
    nms=True,        
    half=False        
)
