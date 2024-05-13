import os
from PIL import Image
from tqdm import tqdm


def Crop_panicleP_Pad_aspectratio(Folder_Raw_Img, Folder_Infer_TXT_position, Folder_Output, target_ratio=1.33333):
    '''
    :param Folder_Raw_Img: 原图文件夹
    :param Folder_Infer_TXT_position: yolov8预测标签文件夹
    :param Folder_Output: 结果图片保存路径保存文件夹
    :param target_ratio: 目标高宽比，根据需求修改
    :return: 将图片根据yolov8预测文件裁剪后，再将高宽比pad到指定数值并保存到新文件夹
    '''
    for file in tqdm(os.listdir(Folder_Raw_Img)):
        coordinate = (0, 0, 0, 0)
        if file.endswith('.jpg'):
            img = Image.open(os.path.join(Folder_Raw_Img, file))  # 打开原图
            image_name = os.path.splitext(os.path.basename(os.path.join(Folder_Raw_Img, file)))[0]  # 去除后缀的图片名
            W, H = img.size  # 图片的高和宽像素值

            txt_file = os.path.join(Folder_Infer_TXT_position, os.path.splitext(file)[0] + '.txt')  # 对应的txt标签文件
            if os.path.exists(txt_file):
                with open(txt_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        line = lines[-1]  # 只读取最后一行
                        data = line.split()
                        Rate_x, Rate_y, Rate_w, Rate_h = float(data[1]), float(data[2]), float(data[3]), float(data[4])
                        x1 = W * Rate_x - 0.5 * W * Rate_w - 50
                        x2 = W * Rate_x + 0.5 * W * Rate_w + 50
                        y1 = H * Rate_y - 0.5 * H * Rate_h - 50
                        y2 = H * Rate_y + 0.5 * H * Rate_h + 50
                        coordinate = (x1, y1, x2, y2)

            padded_img = img.crop(coordinate)  # 直接裁剪子图，不添加填充

            # 计算原图像的宽高比
            original_ratio = padded_img.height / padded_img.width
            # 计算新的宽度和高度
            if original_ratio < target_ratio:
                # 如果原图像的宽高比小于目标宽高比，则新图像的高度为原图像的高度，宽度为原图像宽度的目标比例
                new_width = padded_img.width
                new_height = int(padded_img.width * target_ratio)
            else:
                # 如果原图像的宽高比大于目标宽高比，则新图像的宽度为原图像宽度的目标比例，高度为原图像高度与新宽度的商
                new_width = int(padded_img.height / target_ratio)
                new_height = padded_img.height

            new_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))   # 创建一个新图像，大小为新的宽度和高度，背景颜色为白色

            # 计算图像在新图像中的位置
            x = (new_width - padded_img.width) // 2
            y = (new_height - padded_img.height) // 2

            new_image.paste(padded_img, (x, y))   # 将原图像粘贴到新图像中

            # 保存新图像
            new_image.save(os.path.join(Folder_Output, f"{image_name}.png"))

if __name__ == '__main__':
    Crop_panicleP_Pad_aspectratio('', '', '', 1.33333)
