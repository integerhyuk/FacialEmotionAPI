import cv2
import os

input_folder = './RAF-DB/train/6'
output_folder = './RAF-DB/Resized/6'


for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image = cv2.imread(os.path.join(input_folder, filename))

        # 修改图像分辨率
        resized_image = cv2.resize(image, (48, 48))
        grey_image = cv2.cvtColor(resized_image,cv2.COLOR_BGR2GRAY)

        # 修改图像质量
        jpeg_file = os.path.join(output_folder, filename)
        cv2.imwrite(jpeg_file, grey_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])