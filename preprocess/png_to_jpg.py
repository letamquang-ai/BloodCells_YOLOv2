import os

img_dir = '../dataset/Training/Images'
for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_name)
    os.rename(img_path, img_path[:-3] + 'jpg')

img_dir = '../dataset/Testing/Images'
for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_name)
    os.rename(img_path, img_path[:-3] + 'jpg')

img_dir = '../dataset/Validation/Images'
for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_name)
    os.rename(img_path, img_path[:-3] + 'jpg')
