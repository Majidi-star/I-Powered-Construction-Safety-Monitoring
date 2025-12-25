import os
import glob
from ultralytics import YOLO
import cv2

# --- تنظیمات ---
# پوشه‌ای که نتایج تمرین در آنجا هستند
RUNS_DIR = os.path.join("runs", "detect")
# حداقل درصد اطمینان برای اینکه یک شیء تشخیص داده شود (مثلاً ۴۰ درصد)
CONFIDENCE_THRESHOLD = 0.4

def get_latest_best_model():
    """
    پیدا کردن خودکار جدیدترین فایل best.pt
    این تابع بین تمام پوشه‌های train می‌گردد و آخرین مدلی که ساخته شده را پیدا می‌کند.
    """
    if not os.path.exists(RUNS_DIR):
        print(f"❌ مسیر {RUNS_DIR} پیدا نشد. آیا مدل را آموزش داده‌اید؟")
        return None
    
    # پیدا کردن تمام پوشه‌هایی که با 'train' شروع می‌شوند
    folders = glob.glob(os.path.join(RUNS_DIR, "train*"))
    if not folders:
        print("❌ هیچ پوشه تمرینی پیدا نشد.")
        return None
    
    # مرتب‌سازی بر اساس زمان تغییر برای پیدا کردن آخرین پوشه ایجاد شده
    latest_folder = max(folders, key=os.path.getmtime)
    best_pt_path = os.path.join(latest_folder, "weights", "best.pt")
    
    if os.path.exists(best_pt_path):
        print(f"✅ جدیدترین مدل یافت شد در: {best_pt_path}")
        return best_pt_path
    else:
        print(f"❌ فایل best.pt در پوشه {latest_folder} پیدا نشد.")
        return None

def run_inference(image_path):
    """
    اجرای تست روی یک تصویر مشخص
    """
    # ۱. پیدا کردن و بارگذاری مدل
    model_path = get_latest_best_model()
    if not model_path:
        return

    print("⏳ در حال بارگذاری مدل...")
    model = YOLO(model_path)

    # ۲. بررسی وجود تصویر ورودی
    if not os.path.exists(image_path):
        print(f"❌ تصویر ورودی در مسیر '{image_path}' پیدا نشد.")
        return

    print(f"🚀 در حال پردازش تصویر: {image_path} ...")

    # ۳. انجام پیش‌بینی (Inference)
    results = model.predict(
        source=image_path,
        save=True,            # ذخیره تصویر خروجی با کادر
        conf=CONFIDENCE_THRESHOLD, # حداقل درصد اطمینان
        project="runs/detect",    # محل ذخیره نتایج تست
        name="inference_results", # نام پوشه خروجی تست‌ها
        exist_ok=True         # اگر پوشه وجود داشت، روی آن بنویسد
    )

    # ۴. نمایش گزارش
    for result in results:
        save_dir = result.save_dir
        boxes = result.boxes
        print("\n📊 --- گزارش تشخیص ---")
        print(f"   تعداد اشیاء پیدا شده: {len(boxes)}")
        
        # نمایش کلاس‌های پیدا شده (اختیاری)
        if len(boxes) > 0:
            names = model.names
            detected_classes = [names[int(cls)] for cls in boxes.cls.tolist()]
            print(f"   اشیاء: {set(detected_classes)}")

        print("-" * 30)
        print(f"✨ تصویر خروجی با موفقیت ذخیره شد در:")
        print(f"📂 {save_dir}")
        
        # باز کردن خودکار تصویر خروجی (فقط در ویندوز)
        try:
            output_image_path = os.path.join(save_dir, os.path.basename(image_path))
            if os.name == 'nt' and os.path.exists(output_image_path):
                 os.startfile(output_image_path)
        except Exception:
             pass

if __name__ == "__main__":
    # =========================================
    # 👇👇👇 مسیر تصویر خود را اینجا وارد کنید 👇👇👇
    # می‌توانید یک عکس از اینترنت دانلود کنید و در پوشه پروژه بگذارید
    TEST_IMAGE = "sample_test.jpg" 
    # =========================================
    
    # اجرای برنامه
    run_inference(TEST_IMAGE)