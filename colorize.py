import sys
import os

# إضافة المسار الحالي لضمان عثور بايثون على مجلد deoldify المحلي
sys.path.append(os.getcwd())

try:
    # سنستورد الدوال مباشرة دون الدخول في تعقيدات الـ device
    from deoldify.visualize import get_image_colorizer
    import torch
    import warnings

    # تجاهل التحذيرات
    warnings.filterwarnings("ignore", category=UserWarning)

    # 1. البحث عن الصورة
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_path = next((f for f in os.listdir('.') if f.lower().endswith(valid_extensions) and f != "result.jpg" and not f.startswith('.')), None)

    if not image_path:
        print("❌ خطأ: لم يتم العثور على أي صورة في المستودع!")
        sys.exit(1)

    print(f"🚀 البدء بتلوين الصورة باستخدام DeOldify: {image_path}")

    # 2. تحميل المحرك الفني (سيعمل تلقائياً على CPU)
    # سيقوم النظام بتحميل ملف الأوزان تلقائياً (حوالي 800MB)
    colorizer = get_image_colorizer(artistic=True)

    # 3. معالجة الصورة
    # render_factor=35: دقة ممتازة
    result_img = colorizer.get_transformed_image(
        str(image_path), 
        render_factor=35, 
        post_process=True
    )

    # 4. حفظ النتيجة
    result_img.save("result.jpg")
    
    print("✅ تم التلوين بنجاح باهر!")

except Exception as e:
    print(f"❌ حدث خطأ: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
