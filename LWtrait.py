import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

results = []
lower_RGB, upper_RGB = (26, 109, 159), (134, 189, 214)  # 定义RGB阈值
# lower_red, upper_red = np.array([138, 27, 32]), np.array([248, 100, 108])  # 定义标定物的rgb颜色范围
lower_red, upper_red = np.array([105, 0, 15]), np.array([255, 100, 255])  # 定义标定物的rgb颜色范围
real_radius_mm = 25  # 已知红色实心圆在现实世界中的大小（单位：mm）
# rate_thre = 1.8  # 代表每个籽粒的长宽比阈值（并非均值，是对计算均值前的小均值进行筛选），小于此值就不能被记入结果

# 定义一个函数，根据序号生成两位字母
def get_two_letters(j):
    # 如果序号小于26，直接返回一个大写字母
    if j < 26:
        return chr(ord('A') + j)
    # 否则，返回两个大写字母，第一个是商，第二个是余数
    else:
        return chr(ord('A') + j // 26 - 1) + chr(ord('A') + j % 26)


def get_kernalLW(A_folder, Uncut_img_folder, B_folder, A, B, output_file):
    '''
    :param A_folder: # 原图文件夹
    :param Uncut_img_folder: # 带有标定物的图片文件夹
    :param B_folder: # v7tiny预测的标注文件(.txt)
    :param A: # 分割后每张图片的子图保存路径文件夹
    :param B: # 保存处理后子文件夹的文件夹
    :param output_file: # 结果数据保存文件
    :return: 返回粒长粒宽到指定表格中
    '''
    for file in tqdm(os.listdir(A_folder)):
        rectangles = []  # 此图片所有标注框的坐标信息
        if file.endswith('.jpg'):
            img = Image.open(os.path.join(A_folder, file))  # 打开原图
            img_uncut = cv2.imread(os.path.join(Uncut_img_folder, file))

            # 进行距离矫正，找到合适的像素对真实世界转换比率
            img_rgb = cv2.cvtColor(img_uncut, cv2.COLOR_BGR2RGB)  # 将图像转换成RGB颜色空间
            mask_uncut = cv2.inRange(img_rgb, lower_red, upper_red)  # 根据颜色范围进行红色实心圆的掩码
            kernal = np.ones((7, 7), np.uint8)  # 定义结构元素（kernal）的大小和形状，这里我们使用7x7的矩形结构元素
            img_close = cv2.morphologyEx(mask_uncut, cv2.MORPH_CLOSE, kernal)  # 使用闭运算（close）来消除小物体或者去除物体中的小洞
            contours, _ = cv2.findContours(img_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓

            # 假设只有一个红色实心圆，选择最大的轮廓
            pixels_per_mm = None
            if len(contours) > 0:
                max_contour = max(contours, key=cv2.contourArea)
                (x, y), radius = cv2.minEnclosingCircle(max_contour)
                radius_in_pixels = radius  # 红色实心圆的半径（像素）
                pixels_per_mm = real_radius_mm / radius_in_pixels  # 计算每个像素对应的实际尺寸（mm）

            image_name = os.path.splitext(os.path.basename(os.path.join(A_folder, file)))[0]  # 去除后缀的图片名
            W, H = img.size  # 图片的高和宽像素值
            txt_file = os.path.join(B_folder, os.path.splitext(file)[0] + '.txt')  # 对应的txt标签文件
            if os.path.exists(txt_file):
                with open(txt_file, 'r') as f:
                    for line in f:
                        data = line.split()
                        Rate_x, Rate_y, Rate_w, Rate_h = float(data[1]), float(data[2]), float(data[3]), float(data[4])
                        x1 = W * Rate_x - 0.5 * W * Rate_w
                        x2 = W * Rate_x + 0.5 * W * Rate_w
                        y1 = H * Rate_y - 0.5 * H * Rate_h
                        y2 = H * Rate_y + 0.5 * H * Rate_h
                        coordinate = (x1, y1, x2, y2)
                        rectangles.append(coordinate)

                    # 指定目录，如果不存在就创建
                    directory = os.path.join(A, image_name)
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    # 遍历矩形列表，复制出子图，保存到文件夹中，按照图片名加上英文序列号命名
                    for i, rectangle in enumerate(rectangles):
                        padded_img = ImageOps.expand(img.crop(rectangle), border=(15, 15, 15, 15), fill=1)  # 复制子图并添加 padding
                        padded_img.save(os.path.join(directory, f"{image_name}{get_two_letters(i)}.jpg"))  # 保存子图到文件夹中，拼接子图的完整路径

            sub_save_folder = os.path.join(B, image_name)
            if not os.path.exists(sub_save_folder):
                os.makedirs(sub_save_folder)
            List_chang = []
            List_kuan, List_kuanT5 = [], []
            List_ratio = []
            List_None = []
            sub_folder_path = os.path.join(A, image_name)

            for sub_filename in os.listdir(sub_folder_path):
                sub_path = os.path.join(sub_folder_path, sub_filename)
                if sub_filename.endswith('.jpg'):
                    # 读取图像
                    img = cv2.imread(os.path.join(sub_folder_path, sub_filename))
                    # cv2.imshow('原图', img)
                    # cv2.waitKey(25)

                    # RGB阈值分割
                    mask = cv2.inRange(img, lower_RGB, upper_RGB)
                    # cv2.imshow('阈值分割', mask)
                    # cv2.waitKey(25)

                    # 只保留最大目标
                    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    max_area = 0
                    max_contour = None
                    for contour in contours:
                        area = cv2.contourArea(contour)

                        if area > max_area:
                            max_area = area
                            max_contour = contour
                    mask.fill(0)
                    if max_contour is not None:
                        cv2.drawContours(mask, [max_contour], 0, 255, -1)

                    # 填充空洞
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    # cv2.imshow('空洞填充', mask)
                    # cv2.waitKey(25)

                    # 计算最小外接矩形并绘制
                    try:
                        rect = cv2.minAreaRect(max_contour)
                    except cv2.error:
                        continue
                    # rect = cv2.minAreaRect(max_contour)
                    box = cv2.boxPoints(rect)

                    # img.save(os.path.join(sub_save_folder, sub_filename))
                    w, h = rect[1]
                    rate = max(w, h) / min(w, h)
                    if 1.7 < rate < 6.4 and 49.92 < max(w, h) < 135.20 and 14.04 < min(w, h) < 42.64:
                        List_chang.append(max(w, h))
                        List_kuan.append(min(w, h))
                        List_ratio.append(max(w, h) / min(w, h))
                        box = box.astype(int)
                        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
                        pil_img = Image.fromarray(img)
                        # cv2.imshow('Final Image', img)
                        # cv2.waitKey(25)
                        pil_img.save(os.path.join(sub_save_folder, sub_filename))
            if len(List_chang) >= 3:
                # # 遍历List_chang，去除一个最大值和一个最小值
                # List_chang.remove(max(List_chang))
                # List_chang.remove(min(List_chang))
                #
                # # 去除List_kuan和List_ratio对应位置的值
                # List_kuan.pop(List_chang.index(max(List_chang)))
                # List_ratio.pop(List_chang.index(max(List_chang)))
                # List_kuan.pop(List_chang.index(min(List_chang)))
                # List_ratio.pop(List_chang.index(min(List_chang)))
                #
                # # 遍历List_kuan，去除一个最大值和一个最小值
                # List_kuan.remove(max(List_kuan))
                # List_kuan.remove(min(List_kuan))
                #
                # # 去除List_chang和List_ratio对应位置的值
                # List_chang.pop(List_kuan.index(max(List_kuan)))
                # List_ratio.pop(List_kuan.index(max(List_kuan)))
                # List_chang.pop(List_kuan.index(min(List_kuan)))
                # List_ratio.pop(List_kuan.index(min(List_kuan)))
                #
                # # 遍历List_ratio，去除一个最大值和一个最小值
                # List_ratio.remove(max(List_ratio))
                # List_ratio.remove(min(List_ratio))
                #
                # # 去除List_chang和List_kuan对应位置的值
                # List_chang.pop(List_ratio.index(max(List_ratio)))
                # List_kuan.pop(List_ratio.index(max(List_ratio)))
                # List_chang.pop(List_ratio.index(min(List_ratio)))
                # List_kuan.pop(List_ratio.index(min(List_ratio)))
                # List_kuan_max_5 = sorted(List_kuan, reverse=True)[:5]
                # List_ratio_min_5 = sorted(List_ratio)[:5]
                A_chang = sum(List_chang) / len(List_chang)
                A_kuan = sum(List_kuan) / len(List_kuan)
                A_ratio = sum(List_ratio) / len(List_ratio)
                # A_kuant5 = sum(List_kuan_max_5) / 5
                # A_ratiof5 = sum(List_ratio_min_5) / 5
                real_A_chang = A_chang * pixels_per_mm
                real_A_kuan = A_kuan * pixels_per_mm
                # real_A_kuant5 = A_kuant5 * Exange_S
                # results.append([image_name, real_A_chang, real_A_kuan, real_A_kuant5, A_ratio, A_ratiof5])
                results.append([image_name, real_A_chang, real_A_kuan, A_ratio, pixels_per_mm])
                # print(f"文件{image_name}已经计算完成  粒长:{real_A_chang}  粒宽：{real_A_kuan}   粒宽t5：{real_A_kuant5}  长宽比:{A_ratio} 长宽比f5：{A_ratiof5}")
                print(f"{image_name}  粒长:{real_A_chang}  粒宽：{real_A_kuan}   长宽比:{A_ratio}   单像素对应毫米：{pixels_per_mm}")
            else:
                List_None.append(image_name)
    Num_None = len(List_None)
    print(Num_None)
    df = pd.DataFrame(results, columns=['文件名', '粒长', '粒宽', '长宽比', '单像素对应毫米'])
    df.to_excel(output_file, index=False)

if __name__ == '__main__':
    get_kernalLW(r'', r'', r'', r'', r'', r'')