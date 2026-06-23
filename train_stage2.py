from ultralytics import YOLO

# ================= 核心配置 =================
DATASET_YAML_PATH = "Coffee Detection.v3i.yolov8/data.yaml"
PRETRAINED_WEIGHTS_PATH = "runs/detect/runs/yolov8s_synth26_train2/weights/best.pt"

IMGSZ = 768
DEVICE = "mps"  # 苹果 Apple Silicon 使用 mps
BATCH = 16
WORKERS = 8
AMP = True
# ============================================

def main():
    # 📦 加载之前训练好的 best.pt 模型继续微调
    print(f"📦 正在加载预训练权重: {PRETRAINED_WEIGHTS_PATH}")
    model = YOLO(PRETRAINED_WEIGHTS_PATH)

    # ================== 1. 开始训练 ==================
    print(f"🚀 开始微调训练 (v4)，使用数据集: {DATASET_YAML_PATH}")
    model.train(
        data=DATASET_YAML_PATH,
        imgsz=IMGSZ,
        epochs=50,             # 微调轮数设为 50
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        
        pretrained=True,
        optimizer="SGD",
        lr0=0.001,             # 学习率降低（0.01 -> 0.001），适合微调
        lrf=0.1,               # 最终学习率系数调整
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=0,       #  取消 warmup（已经是训练过的模型，不需要预热）
        patience=50,
        cos_lr=True,

        # ------ 数据增强（针对微调：锁死颜色，极低几何形变，关闭马赛克） ------
        hsv_h=0,               #  彻底锁死色相
        hsv_s=0.1,             #  低饱和度扰动
        hsv_v=0.1,             #  极低明度扰动
        degrees=5.0,
        translate=0.05,        # 降低平移幅度
        scale=0.2,             # 降低缩放幅度
        shear=0.0,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0,              # 完全关闭拼接增强（贴近真实数据分布）
        close_mosaic=10,
        mixup=0.0,
        auto_augment=None,
        erasing=0.0,

        # ------ 训练控制 ------
        amp=AMP,
        cache=False,
        verbose=True,
        plots=True,
        project="runs",
        name="yolov8s_train",  
    )

    # ================== 2. 测试集评估 ==================
    print("\n📊 微调完成！正在独立测试集(Test)上评估最终指标...")
    try:
        model.val(data=DATASET_YAML_PATH, imgsz=IMGSZ, device=DEVICE, split="test")
        print(f"✅ 测试评估完成！结果已保存在 runs/val 目录下。")
    except Exception as e:
        print(f"⚠️ Test 评估跳过或失败，原因: {e}")

if __name__ == "__main__":
    main()