import cv2
import numpy as np
import urllib.request
import os
import sys

# روابط مباشرة ومستقرة تماماً
prototxt_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/f930e6e5a593322b79a556396e944b25451e0655/colorization_deploy_v2.prototxt"
model_url = "https://github.com/richzhang/colorization/raw/master/colorization/models/colorization_release_v2.caffemodel"
points_url = "https://github.com/richzhang/colorization/raw/master/colorization/resources/pts_in_hull.npy"

def download_file(url, filename):
    print(f"📥 جاري جلب: {filename}...")
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, filename)

try:
    # تنزيل الملفات اللازمة
    download_file(prototxt_url, "color.prototxt")
    download_file(model_url, "color.caffemodel")
    download_file(points_url, "pts.npy")

    # البحث عن صورة في المستودع
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_path = None
    for file in os.listdir('.'):
        if file.lower().endswith(valid_extensions) and file != "result.jpg":
            image_path = file
            break

    if not image_path:
        print("❌ لم أجد أي صورة (jpg/png) في المستودع!")
        sys.exit(1)

    print(f"🎨 سأقوم بتلوين: {image_path}")

    # محرك التلوين
    net = cv2.dnn.readNetFromCaffe("color.prototxt", "color.caffemodel")
    pts = np.load("pts.npy")
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    net.getLayer(class8).blobs = [pts.transpose().reshape(1, 313, 1, 1).astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    image = cv2.imread(image_path)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

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

    cv2.imwrite("result.jpg", colorized)
    print("✅ تم بنجاح! النتيجة محفوظة في result.jpg")

except Exception as e:
    print(f"❌ حدث خطأ غير متوقع: {e}")
    sys.exit(1)
