from pathlib import Path
from PIL import Image
from pillow_heif import register_heif_opener

IN_DIR = Path("heic_img")
OUT_DIR = Path("jpg_img")
QUALITY = 92  

def main():
    register_heif_opener()

    if not IN_DIR.exists() or not IN_DIR.is_dir():
        raise SystemExit(f"找不到输入文件夹：{IN_DIR.resolve()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(IN_DIR.glob("*.heic")) + sorted(IN_DIR.glob("*.HEIC"))
    if not files:
        raise SystemExit(f"{IN_DIR.resolve()} 下没有 .heic/.HEIC 文件")

    ok = 0
    for src in files:
        try:
            with Image.open(src) as im:
                im = im.convert("RGB")
                dst = OUT_DIR / (src.stem + ".jpg")
                im.save(dst, format="JPEG", quality=QUALITY, optimize=True)
            ok += 1
            print(f"OK  {src.name} -> {dst.name}")
        except Exception as e:
            print(f"FAIL {src.name} ({e})")

    print(f"完成：成功 {ok}/{len(files)}，输出：{OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()