import sys
import os
import urllib.request

# إضافة المسار الحالي
sys.path.append(os.getcwd())

def download_weights():
    """تحميل ملف أوزان النموذج يدوياً لضمان وجوده في المسار الصحيح"""
    weights_url = "https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth"
    weights_dir = "models"
    weights_path = os.path.join(weights_dir, "ColorizeArtistic_gen.pth")
    
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
        
    if not os.path.exists(weights_path):
        print(f"📦 جاري تحميل ملف الأوزان الضخم (800MB)... قد يستغرق هذا من 2 إلى 5 دقائق")
        try:
            urllib.request.urlretrieve(weights_url, weights_path)
            print("✅ اكتمل تحميل الأوزان بنجاح!")
        except Exception as e:
            print(f"❌ فشل تحميل الأوزان: {e}")
            sys.exit(1)

try:
    # 1. تحميل الأوزان أولاً
    download_weights()

    import torch
    
    # --- الحل السحري للخطأ الجديد ---
    # نخبر Torch بأن يثق في دالة slice التي يطلبها ملف الأوزان القديم
    torch.serialization.add_safe_globals([slice])
    # --------------------------------

    from deoldify.visualize import get_image_colorizer
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    # 2. البحث عن الصورة
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_path = next((f for f in os.listdir('.') if f.lower().endswith(valid_extensions) and f != "result.jpg" and not f.startswith('.')), None)

    if not image_path:
        print("❌ خطأ: لم يتم العثور على أي صورة!")
        sys.exit(1)

    print(f"🚀 البدء بتلوين الصورة باستخدام DeOldify: {image_path}")

    # 3. تحميل المحرك
    colorizer = get_image_colorizer(artistic=True)

    # 4. المعالجة
    result_img = colorizer.get_transformed_image(
        str(image_path), 
        render_factor=35, 
        post_process=True
    )

    # 5. الحفظ
    result_img.save("result.jpg")
    print("✅ تم التلوين بنجاح باهر!")

except Exception as e:
    print(f"❌ حدث خطأ في المحرك: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
