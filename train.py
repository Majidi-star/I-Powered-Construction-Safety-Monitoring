import os
import glob
from ultralytics import YOLO

# --- تنظیمات ---
DATASET_DIR = "dataset"
KAGGLE_SLUG = "snehilsanyal/construction-site-safety-image-dataset-roboflow"
RUNS_DIR = os.path.join("runs", "detect")

def get_latest_checkpoint():
    """پیدا کردن جدیدترین پوشه تمرین و فایل last.pt"""
    if not os.path.exists(RUNS_DIR):
        return None
    
    # پیدا کردن تمام پوشه‌هایی که با 'train' شروع می‌شوند
    folders = glob.glob(os.path.join(RUNS_DIR, "train*"))
    if not folders:
        return None
    
    # مرتب‌سازی بر اساس زمان تغییر برای پیدا کردن آخرین پوشه
    latest_folder = max(folders, key=os.path.getmtime)
    checkpoint_path = os.path.join(latest_folder, "weights", "last.pt")
    
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    return None

def prepare_dataset():
    """بررسی وجود دیتاست و دانلود از Kaggle"""
    if not os.path.exists(DATASET_DIR):
        print("🚀 دیتاست یافت نشد. در حال دانلود...")
        try:
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(KAGGLE_SLUG, path=DATASET_DIR, unzip=True)
            print("✅ دانلود موفقیت‌آمیز بود.")
        except Exception as e:
            print(f"❌ خطا در دانلود: {e}")
            return False
    return True

def start_training():
    """شروع یا ادامه هوشمند تمرین"""
    last_pt = get_latest_checkpoint()
    
    if last_pt:
        print(f"🔄 پیدا شد! در حال ادامه تمرین از جدیدترین چک‌پوینت: {last_pt}")
        model = YOLO(last_pt)
        # استفاده از resume=True باعث می‌شود در همان پوشه قبلی ادامه دهد و پوشه جدید نسازد
        model.train(resume=True)
    else:
        print("🆕 هیچ تمرین قبلی پیدا نشد. شروع آموزش جدید...")
        model = YOLO('yolov8n.pt')
        model.train(
            data='data.yaml',
            epochs=30,
            imgsz=640,
            device=0,
            workers=2, # جلوگیری از خطای Paging File
            batch=16,
            name='train' # نام پایه پوشه ذخیره‌سازی
        )

if __name__ == "__main__":
    if prepare_dataset():
        start_training()