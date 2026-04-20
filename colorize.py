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
    """تحميل أوزان المحركين (Artistic & Stable) بالروابط الجديدة والمؤكدة"""
    # الروابط التي جلبتها وتأكدت منها
    models_info = {
        "ColorizeArtistic_gen.pth": "https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth",
        "ColorizeStable_gen.pth": "https://www.dropbox.com/s/axsd2g85uyixaho/ColorizeStable_gen.pth?dl=1"
    }
    
    if not os.path.exists("models"):
        os.makedirs("models")
        
    # إعداد محمل الروابط ليتخطى حماية البوتات (مهم لـ Dropbox)
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    for filename, url in models_info.items():
        path = os.path.join("models", filename)
        name = "المحرك الفني" if "Artistic" in filename else "المحرك المستقر"
        
        if not os.path.exists(path):
            print(f"📦 جاري تحميل {name} (800MB)...")
            try:
                urllib.request.urlretrieve(url, path)
                print(f"✅ تم تحميل {name} بنجاح")
            except Exception as e:
                print(f"❌ فشل تحميل {name}: {e}")
                sys.exit(1)
        else:
            print(f"✔ {name} موجود مسبقاً، تخطي التحميل.")

def get_optimal_render_factor(image_path):
    """الذكاء الاصطناعي يقرر الرقم المناسب بناءً على أبعاد الصورة"""
    with Image.open(image_path) as img:
        width, height = img.size
    pixels = width * height
    if pixels > 2000000: return 35
    elif pixels > 1000000: return 28
    else: return 21

def colorize_step(is_artistic, image_path, rf):
    """تشغيل المحرك ثم تنظيف الذاكرة تماماً"""
    from deoldify.visualize import get_image_colorizer
    
    # اختيار اسم الوزن المناسب للمحرك
    weights_name = "ColorizeArtistic_gen" if is_artistic else "ColorizeStable_gen"
    
    # تحميل المحرك
    colorizer = get_image_colorizer(artistic=is_artistic)
    
    print(f"🎨 معالجة {'Artistic' if is_artistic else 'Stable'} (RF: {rf})")
    result = colorizer.get_transformed_image(str(image_path), render_factor=rf, post_process=True)
    
    # تفريغ الرام (مهم جداً للـ 7GB)
    del colorizer
    gc.collect()
    if torch.cuda.is_available(): 
        torch.cuda.empty_cache()
    
    return result

try:
    # 1. إعداد البيئة وتحميل الأوزان (بالروابط الجديدة)
    download_weights()
    
    # حل مشكلة الأمان في Torch 2.6
    torch.serialization.add_safe_globals([slice])
    warnings.filterwarnings("ignore", category=UserWarning)

    # 2. البحث عن الصورة المراد تلوينها
    valid_extensions = ('.jpg', '.jpeg', '.png')
    img_name = next((f for f in os.listdir('.') if f.lower().endswith(valid_extensions) and "result" not in f and not f.startswith('.')), None)

    if not img_name:
        print("❌ خطأ: لم يتم العثور على أي صورة في المستودع!")
        sys.exit(1)

    # 3. القرار التلقائي للـ Render Factor
    rf = get_optimal_render_factor(img_name)
    print(f"🚀 الصورة المكتشفة: {img_name}")
    print(f"🤖 تقرر استخدام Render Factor: {rf}")

    # 4. المعالجة المزدوجة (محرك تلو الآخر لتوفير الرام)
    print("⏳ جاري تشغيل المحرك الأول (الفني)...")
    img_art = colorize_step(True, img_name, rf)
    
    print("⏳ جاري تشغيل المحرك الثاني (المستقر)...")
    img_stable = colorize_step(False, img_name, rf)

    # 5. دمج النتائج بنسبة 50/50
    print("🧪 جاري دمج النتائج للحصول على توازن الألوان المثالي...")
    final_result = Image.blend(img_art, img_stable, alpha=0.5)

    # 6. حفظ النتيجة النهائية والنسخ الإضافية
    final_result.save("result.jpg")
    img_art.save("result_artistic.jpg")
    img_stable.save("result_stable.jpg")
    
    print("✅ تم التلوين والدمج بنجاح باهر يا لموشي! تفقد الـ Artifacts.")

except Exception as e:
    print(f"❌ حدث خطأ في المحرك المطور: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
