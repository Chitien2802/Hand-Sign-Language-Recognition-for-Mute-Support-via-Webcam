#Name: Nguyen Chi Tien
#Location: Department of Civil Engineering and Environment, College of Engineering, Myongji University, 116 Myongji-ro, Cheoin-gu, Yongin, Gyeonggy-do 449-728, Korea.

import os
import cv2
import albumentations as A
import shutil

# input
input_image_folder = r""
input_label_folder = r"d:"

# output
output_image_folder = r"d:"
output_label_folder = r"d:"

os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_label_folder, exist_ok=True)

# number of images augment each original image
augment_number = 6

#  pipeline augmentation
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=30, p=0.5),
    A.Blur(p=0.1),
    A.Affine(translate_percent=0.05, scale=(0.95, 1.05), rotate=(-15, 15), p=0.5)
])

# Duyệt qua từng ảnh
for filename in os.listdir(input_image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_image_folder, filename)
        image = cv2.imread(image_path)

        name, ext = os.path.splitext(filename)
        label_file = name + ".txt"
        label_path = os.path.join(input_label_folder, label_file)

        if not os.path.exists(label_path):
            print(f" Không tìm thấy label cho ảnh: {filename}")
            continue

        for i in range(augment_number):
            augmented = transform(image=image)['image']
            new_img_name = f"{name}_aug{i+1}{ext}"
            new_label_name = f"{name}_aug{i+1}.txt"

            # Lưu ảnh đã augment
            cv2.imwrite(os.path.join(output_image_folder, new_img_name), augmented)

            # Copy label gốc sang với tên mới
            shutil.copy(label_path, os.path.join(output_label_folder, new_label_name))

print("Augmentation OK.")
