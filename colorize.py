import cv2
import numpy as np
import urllib.request
import os

# الروابط الجديدة والمضمونة للنموذج
prototxt_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/f930e6e5a593322b79a556396e944b25451e0655/colorization_deploy_v2.prototxt"
model_url = "https://github.com/richzhang/colorization/raw/master/colorization/models/colorization_release_v2.caffemodel"
points_url = "https://github.com/richzhang/colorization/raw/master/colorization/resources/pts_in_hull.npy"

# تنزيل الملفات
print("جاري تحميل ملفات الذكاء الاصطناعي...")
urllib.request.urlretrieve(prototxt_url, "color.prototxt")
urllib.request.urlretrieve(model_url, "color.caffemodel")
urllib.request.urlretrieve(points_url, "pts.npy")

# إعداد الشبكة العصبية
net = cv2.dnn.readNetFromCaffe("color.prototxt", "color.caffemodel")
pts = np.load("pts.npy")

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
net.getLayer(class8).blobs = [pts.transpose().reshape(1, 313, 1, 1).astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# تأكد من اسم الصورة التي رفعتها (يجب أن يكون test_page.jpg)
image_path = "test_page.jpg"
if not os.path.exists(image_path):
    # إذا لم يجد الصورة بهذا الاسم، سيبحث عن أي صورة jpg أخرى
    files = [f for f in os.listdir('.') if f.endswith(('.jpg', '.png', '.jpeg'))]
    if files: image_path = files[0]

print(f"جاري معالجة الصورة: {image_path}")
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
print("تم التلوين بنجاح وحفظ الصورة باسم result.jpg")
