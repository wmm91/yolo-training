from ultralytics import YOLO

# ========= 训练配置 =========
DATA_YAML = "Coffee Detection.v2i.yolov8/data.yaml"
IMGSZ = 768
DEVICE = "mps"  # 无 MPS 用 "cpu"
BATCH = 16
WORKERS = 8
EPOCHS = 200
PATIENCE = 50  # 早停耐心值

def main():
    model = YOLO("yolov8s.pt")
    
    model.train(
        data=DATA_YAML,
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        
        # 优化器配置
        pretrained=True,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        patience=PATIENCE,
        cos_lr=True,
        
        # 数据增强
        hsv_h=0.005,      # 色相增强（弱）
        hsv_s=0.3,        # 饱和度增强
        hsv_v=0.2,        # 明度增强
        degrees=5.0,      # 旋转角度
        translate=0.1,    # 平移
        scale=0.4,        # 缩放
        fliplr=0.5,       # 水平翻转
        mosaic=0.3,       # 拼接增强（降低强度）
        close_mosaic=10,  # 提前关闭mosaic
        mixup=0.0,        # 关闭mixup
        
        # 其他
        amp=True,         # 混合精度训练
        plots=True,       # 保存曲线图
        project="runs",
        name="yolov8s_synth26_train",
    )

if __name__ == "__main__":
    main()