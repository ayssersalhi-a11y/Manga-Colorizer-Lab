import cv2
import numpy as np
import os
import sys

try:
    # التأكد من وجود الملفات
    if not os.path.exists("color.prototxt") or not os.path.exists("color.caffemodel") or not os.path.exists("pts.npy"):
        print("❌ الملفات الأساسية مفقودة!")
        sys.exit(1)

    # البحث عن أي صورة
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_path = next((f for f in os.listdir('.') if f.lower().endswith(valid_extensions) and f != "result.jpg"), None)

    if not image_path:
        print("❌ لم يتم العثور على صورة!")
        sys.exit(1)

    print(f"🎨 جاري معالجة: {image_path}")

    # تحميل النموذج
    net = cv2.dnn.readNetFromCaffe("color.prototxt", "color.caffemodel")
    pts = np.load("pts.npy")

    # إصلاح مشكلة الـ Reshape (تغيير طريقة قراءة النقاط لتناسب حجم 626)
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    
    # هنا التعديل: التأكد من تحويل النقاط لشكل (2, 313) ثم (1, 313, 1, 1)
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # قراءة الصورة ومعالجتها
    image = cv2.imread(image_path)
    if image is None:
        print("❌ فشل في قراءة الصورة!")
        sys.exit(1)
        
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # التلوين بالذكاء الاصطناعي
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = (255 * colorized).astype("uint8")

    # حفظ النتيجة
    cv2.imwrite("result.jpg", colorized)
    print("✅ تم التلوين بنجاح! تفقد الـ Artifacts لتحميل صورتك.")

except Exception as e:
    print(f"❌ خطأ تقني: {e}")
    sys.exit(1)
