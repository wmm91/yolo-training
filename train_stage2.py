"""
基于第一阶段得到的best模型继续训练
"""
from pathlib import Path

from ultralytics import YOLO

# ========= 训练配置 =========
DATA_YAML = "Coffee Detection.v3i.yolov8/data.yaml"
PRETRAINED_WEIGHTS = "runs/detect/runs/yolov8s_synth26_train2/weights/best.pt"
IMGSZ = 768
DEVICE = "mps"  # 无 MPS 用 "cpu"
BATCH = 16
WORKERS = 8
EPOCHS = 50


def main():
    project_root = Path(__file__).resolve().parent
    
    # 加载预训练权重
    weights = (project_root / PRETRAINED_WEIGHTS).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"找不到预训练权重: {weights}")
    model = YOLO(str(weights))
    
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
        lr0=0.001,      # 使用更小的学习率（微调）
        lrf=0.1,        # 最终学习率 = lr0 * lrf
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=0,  # 不使用warmup
        patience=50,
        cos_lr=True,
        
        # 数据增强（v4调整：降低颜色和几何增强强度）
        hsv_h=0,          # 锁死色相
        hsv_s=0.1,        # 饱和度微调
        hsv_v=0.1,        # 明度微调
        degrees=5.0,      # 旋转角度
        translate=0.05,   # 平移（降低）
        scale=0.2,        # 缩放（降低）
        fliplr=0.5,       # 水平翻转
        mosaic=0,         # 关闭拼接增强
        mixup=0.0,        # 关闭mixup
        
        # 其他
        amp=True,         # 混合精度训练
        plots=True,       # 保存曲线图
        project="runs",
        name="yolov8s_synth26_cdv4",
    )


if __name__ == "__main__":
    main()