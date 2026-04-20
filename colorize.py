from deoldify import device
from deoldify.device_id import DeviceId
import torch

# إجبار المحرك على استخدام المعالج (CPU) ليتوافق مع مواصفات GitHub Actions المجانية
device.set_ptr_to_memory(DeviceId.CPU)

from deoldify.visualize import *
import os
import sys
import warnings

# تجاهل التحذيرات غير الضرورية لتنظيف سجل التشغيل (Logs)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    # 1. البحث عن الصورة المراد تلوينها في المستودع
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_path = next((f for f in os.listdir('.') if f.lower().endswith(valid_extensions) and f != "result.jpg"), None)

    if not image_path:
        print("❌ خطأ: لم يتم العثور على أي صورة (jpg, png) في المستودع!")
        sys.exit(1)

    print(f"🚀 البدء بتلوين الصورة باستخدام DeOldify: {image_path}")

    # 2. تحميل المحرك الفني (Artistic)
    # ملاحظة: سيقوم الأكسيون بتحميل ملفات الأوزان (Weights) في المرة الأولى تلقائياً
    colorizer = get_image_colorizer(artistic=True)

    # 3. معالجة الصورة
    # render_factor=35 هو توازن ممتاز بين الجودة واستهلاك الرام (7 جيجا المتاحة)
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
    sys.exit(1)
