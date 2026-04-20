import sys
import os
import urllib.request

# إضافة المسار الحالي لضمان عثور بايثون على مجلد deoldify المحلي
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
            # تحميل الملف مع إظهار حالة بسيطة
            urllib.request.urlretrieve(weights_url, weights_path)
            print("✅ اكتمل تحميل الأوزان بنجاح!")
        except Exception as e:
            print(f"❌ فشل تحميل الأوزان من الرابط الأساسي: {e}")
            sys.exit(1)

try:
    # تحميل الأوزان أولاً قبل أي استيراد للمحرك
    download_weights()

    from deoldify.visualize import get_image_colorizer
    import torch
    import warnings

    # تجاهل التحذيرات لتنظيف سجل التشغيل
    warnings.filterwarnings("ignore", category=UserWarning)

    # 1. البحث عن الصورة المراد تلوينها في المستودع
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_path = next((f for f in os.listdir('.') if f.lower().endswith(valid_extensions) and f != "result.jpg" and not f.startswith('.')), None)

    if not image_path:
        print("❌ خطأ: لم يتم العثور على أي صورة (jpg, png) في المستودع!")
        sys.exit(1)

    print(f"🚀 البدء بتلوين الصورة باستخدام DeOldify: {image_path}")

    # 2. تحميل المحرك الفني (Artistic)
    # سيعمل تلقائياً على CPU في بيئة GitHub Actions
    colorizer = get_image_colorizer(artistic=True)

    # 3. معالجة الصورة
    # render_factor=35: توازن ممتاز بين الجودة واستهلاك الذاكرة
    result_img = colorizer.get_transformed_image(
        str(image_path), 
        render_factor=35, 
        post_process=True
    )

    # 4. حفظ النتيجة النهائية
    result_img.save("result.jpg")
    
    print("✅ تم التلوين بنجاح باهر! يمكنك الآن تحميل النتيجة من Artifacts.")

except Exception as e:
    print(f"❌ حدث خطأ في المحرك: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
