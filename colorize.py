import sys
import os
import urllib.request
import torch
import gc 
from PIL import Image
import warnings

# إضافة المسار الحالي لضمان عثور بايثون على مجلد deoldify المحلي
sys.path.append(os.getcwd())

def download_weights():
    """تحميل أوزان المحركين (Artistic & Stable) لضمان العمل بسلام"""
    base_url = "https://data.deepai.org/deoldify/"
    models = {
        "ColorizeArtistic_gen.pth": "المحرك الفني",
        "ColorizeStable_gen.pth": "المحرك المستقر"
    }
    
    if not os.path.exists("models"):
        os.makedirs("models")
        
    for filename, name in models.items():
        path = os.path.join("models", filename)
        if not os.path.exists(path):
            print(f"📦 جاري تحميل {name} (800MB)...")
            try:
                urllib.request.urlretrieve(base_url + filename, path)
                print(f"✅ تم تحميل {name}")
            except Exception as e:
                print(f"❌ فشل تحميل {name}: {e}")
                sys.exit(1)

def get_optimal_render_factor(image_path):
    """الذكاء الاصطناعي يقرر الرقم المناسب بناءً على أبعاد الصورة"""
    with Image.open(image_path) as img:
        width, height = img.size
    pixels = width * height
    if pixels > 2000000: return 35
    elif pixels > 1000000: return 28
    else: return 21

def colorize_step(is_artistic, image_path, rf):
    """تشغيل المحرك ثم الانتحار برمجياً لتفريغ الرام"""
    from deoldify.visualize import get_image_colorizer
    colorizer = get_image_colorizer(artistic=is_artistic)
    
    print(f"🎨 معالجة {'Artistic' if is_artistic else 'Stable'} (RF: {rf})")
    result = colorizer.get_transformed_image(str(image_path), render_factor=rf, post_process=True)
    
    # تنظيف الرام فوراً
    del colorizer
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result

try:
    # 1. إعداد البيئة وتحميل الأوزان (من كودك الأول)
    download_weights()
    
    # حل مشكلة الأمان في Torch 2.6 (من كودك الأول)
    torch.serialization.add_safe_globals([slice])
    warnings.filterwarnings("ignore", category=UserWarning)

    # 2. البحث عن الصورة (من كودك الأول)
    valid_extensions = ('.jpg', '.jpeg', '.png')
    img_name = next((f for f in os.listdir('.') if f.lower().endswith(valid_extensions) and "result" not in f and not f.startswith('.')), None)

    if not img_name:
        print("❌ خطأ: لم يتم العثور على أي صورة!")
        sys.exit(1)

    # 3. القرار التلقائي للـ Render Factor
    rf = get_optimal_render_factor(img_name)
    print(f"🚀 البدء بمعالجة: {img_name} (RF المختار: {rf})")

    # 4. المعالجة المزدوجة (تفريغ الرام بينهما لضمان الـ 7GB)
    img_art = colorize_step(True, img_name, rf)
    img_stable = colorize_step(False, img_name, rf)

    # 5. دمج النتائج (خضرة العشب من Stable + عمق الظلال من Artistic)
    print("🧪 جاري دمج النتائج بنسبة 50/50...")
    final_result = Image.blend(img_art, img_stable, alpha=0.5)

    # 6. حفظ النتيجة النهائية (من كودك الأول)
    final_result.save("result.jpg")
    # نسخ إضافية للمقارنة
    img_art.save("result_artistic.jpg")
    img_stable.save("result_stable.jpg")
    
    print("✅ تم التلوين والدمج بنجاح باهر يا لموشي!")

except Exception as e:
    print(f"❌ حدث خطأ في المحرك المطور: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
