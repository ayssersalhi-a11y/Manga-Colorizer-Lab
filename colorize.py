import sys
import os

# إضافة المسار الحالي لضمان عثور بايثون على مجلد deoldify المحلي
sys.path.append(os.getcwd())

try:
    from deoldify import device
    from deoldify.device_id import DeviceId
    import torch
    
    # إجبار المحرك على استخدام المعالج (CPU)
    device.set_ptr_to_memory(DeviceId.CPU)
    
    from deoldify.visualize import *
    import warnings

    # تجاهل التحذيرات لتنظيف سجل التشغيل
    warnings.filterwarnings("ignore", category=UserWarning)

    # 1. البحث عن الصورة المراد تلوينها
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_path = next((f for f in os.listdir('.') if f.lower().endswith(valid_extensions) and f != "result.jpg" and not f.startswith('.')), None)

    if not image_path:
        print("❌ خطأ: لم يتم العثور على أي صورة (jpg, png) في المستودع!")
        sys.exit(1)

    print(f"🚀 البدء بتلوين الصورة باستخدام DeOldify: {image_path}")

    # 2. تحميل المحرك الفني (Artistic)
    # سيقوم النظام بتحميل ملف الأوزان (Weights) تلقائياً (حوالي 800MB)
    colorizer = get_image_colorizer(artistic=True)

    # 3. معالجة الصورة
    # render_factor=35 هو توازن ممتاز لمواصفات رام GitHub
    result_img = colorizer.get_transformed_image(
        image_path, 
        render_factor=35, 
        post_process=True
    )

    # 4. حفظ النتيجة النهائية
    result_img.save("result.jpg")
    
    print("✅ تم التلوين باحترافية! الملف الناتج هو: result.jpg")

except Exception as e:
    print(f"❌ حدث خطأ في المحرك المطور: {e}")
    # طباعة تفاصيل الخطأ للمساعدة في التصحيح إذا لزم الأمر
    import traceback
    traceback.print_exc()
    sys.exit(1)
