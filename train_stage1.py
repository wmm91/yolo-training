from ultralytics import YOLO

# ================= 核心配置 =================
# 🌟 在这里自定义你的数据集 yaml 文件路径（建议写绝对路径）
# 例如: "/Users/你的用户名/Desktop/Coffee Detection/data.yaml"
DATASET_YAML_PATH = "Coffee Detection.v2i.yolov8/data.yaml" 

IMGSZ = 768
DEVICE = "mps"  # 苹果 Apple Silicon 使用 mps
BATCH = 16
WORKERS = 8
AMP = True
# ============================================

def main():
    model = YOLO("yolov8s.pt")

    # ================== 1. 开始训练 ==================
    print(f"🚀 开始训练模型，使用数据集: {DATASET_YAML_PATH}")
    model.train(
        data=DATASET_YAML_PATH,
        imgsz=IMGSZ,
        epochs=200,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        
        pretrained=True,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        patience=50,
        cos_lr=True,

        # ------ 数据增强（严格保护咖啡杯特征） ------
        hsv_h=0.005,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=5.0,
        translate=0.1,
        scale=0.4,
        shear=0.0,           # 显式关闭剪切形变
        fliplr=0.5,
        flipud=0.0,          # 显式关闭上下翻转
        mosaic=0.3,
        close_mosaic=10,
        mixup=0.0,           # 关闭 mixup
        auto_augment=None,   # 关闭自动增强
        erasing=0.0,         # 显式关闭随机擦除

        # ------ 训练控制 ------
        amp=AMP,
        cache=True,         # 内存若充裕（如 16G 以上）可改为 True 提速
        verbose=True,
        plots=True,          # 自动保存 loss 曲线等图表
        project="runs",
        name="yolov8s_train",
    )

    # ================== 2. 测试集评估 ==================
    print("\n📊 训练完成！正在独立测试集(Test)上评估最终指标...")
    try:
        model.val(data=DATASET_YAML_PATH, imgsz=IMGSZ, device=DEVICE, split="test")
        print(f"✅ 测试评估完成！结果已保存在 runs/val 目录下。")
    except Exception as e:
        print(f"⚠️ Test 评估跳过或失败，原因: {e}")

if __name__ == "__main__":
    main()