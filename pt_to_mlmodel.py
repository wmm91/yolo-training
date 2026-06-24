from ultralytics import YOLO

# 加载模型
# 修改为你的模型路径
model = YOLO('yolov8sbest.pt')

# 导出模型
model.export(
    format='coreml', 
    imgsz=768,       
    nms=True,        
    half=False        
)
