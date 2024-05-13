import cv2
import heapq
import numpy as np
import os
import sys
import pandas as pd
from PIL import Image
from skimage.morphology import skeletonize
from tqdm import tqdm


sys.setrecursionlimit(1000000000)
# lower_red, upper_red= np.array([138, 27, 32]), np.array([248, 100, 108])  # 定义标定物的rgb颜色范围
lower_red, upper_red = np.array([105, 0, 15]), np.array([255, 100, 255])  # 定义标定物的rgb颜色范围
lower, upper = (0, 20, 20), (255, 255, 255)  # 指定稻穗的bgr颜色范围
HOLE_AREA_THRESHOLD = 1000  # 填补空洞阈值
real_radius_mm = 25  # 已知红色实心圆在现实世界中的大小（单位：mm）
area_threshold = 37000  # 固定面积阈值
top_count, left_count, right_count = 30, 20, 20  # 计算顶部，左侧，右侧的端点个数

# input_folder = r"H:\NewGWAs_img\916\cut"  # 裁剪后稻穗图片文件夹
# Uncut_img_folder = r"H:\NewGWAs_img\916\raw_img"  # 未经过裁剪 带有标定物的原图
# output_folder = r"H:\test\919\Panicle_length\919panicle_length"  # 保存稻穗主路径图像的文件夹
# output_file = r"H:\test\919\Panicle_length\919panicle_length.xlsx"  # 存放穗长数据的表格
results = []  # 存放穗长数据的列表


def get_panicle_length(input_folder, Uncut_img_folder, output_folder, output_file):
    def astar(start, goal, h):  # A*算法
        frontier = [(0, start)]
        came_from = {}
        cost_so_far = {start: 0}
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break

            for neighbor in neighbors(current):
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + h(neighbor, goal)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

                    # 重建路径
        path = []
        current = goal
        while current != start:
            path.append(current)
            if current not in came_from:  # 检查came_from字典中是否存在当前键值
                break  # 如果不存在，跳出循环，防止KeyError
            current = came_from[current]
        path.append(start)
        path.reverse()

        return path if path else None  # 如果没有找到路径，返回None

    # 计算启发式估价函数h
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def neighbors(point):  # 获取一个点的所有相邻点
        y, x = point
        result = []
        for i in range(y - 1, y + 2):
            for j in range(x - 1, x + 2):
                if not (i == y and j == x) and skeleton[i, j] != 0:
                    result.append((i, j))
        return result
    for filename in tqdm(os.listdir(input_folder)):  # 遍历文件夹中的所有图片
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # 读取图片
            img = cv2.imread(os.path.join(input_folder, filename))
            img_uncut = cv2.imread(os.path.join(Uncut_img_folder, filename))
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

                # print(filename, pixels_per_mm)

            # cv2.drawContours(black_image, [max_contour], 0, 255, -1)
            # cv2.imwrite(os.path.join(output_folder, filename), black_image)

            mask_img = cv2.inRange(img, lower, upper)  # 阈值分割
            roi = cv2.bitwise_and(img, img, mask=mask_img)  # 提取感兴趣区域
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # 将感兴趣区域转为为hsv空间
            _, _, s = cv2.split(hsv)  # 提取S分量图
            _, thresh = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津法（Otsu）阈值分割
            # cv2.imwrite(os.path.join(output_folder, filename), thresh)


            # 滤除面积小于阈值的部分
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < area_threshold:
                    cv2.drawContours(thresh, [contour], -1, 0, -1)
            # cv2.imwrite(os.path.join(output_folder, filename), thresh)

            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            for i in range(len(contours)):
                if hierarchy[0][i][3] != -1:  # 检测当前轮廓是否为孔洞
                    hole_area = cv2.contourArea(contours[i])  # 计算孔洞的面积

                    if hole_area < HOLE_AREA_THRESHOLD:  # 如果孔洞的面积小于阈值
                        cv2.drawContours(thresh, [contours[i]], 0, 255, -1)  # 进行像素填充处理
            kernel = np.ones((9, 9), np.uint8)  # 定义结构元素（kernal）的大小和形状，这里我们使用9*9的矩形结构元素
            dilated = cv2.dilate(thresh, kernel, iterations=2)
            # cv2.imwrite(os.path.join(output_folder, filename), dilated)

            # 提取图像骨架并
            skeleton = skeletonize(dilated)
            skeleton = np.uint8(skeleton) * 255
            skeleton = np.pad(skeleton, ((1, 1), (1, 1)), mode='constant', constant_values=0)
            # cv2.imwrite(os.path.join(output_folder, filename), skeleton)

            # 获取骨架中所有点
            points = []
            for S_point in zip(*np.where(skeleton == 255)):
                points.append(S_point)

            # 寻找所有端点
            end_points = []
            for point in points:
                y, x = point
                num_neighbors = 0
                for i in range(y - 1, y + 2):
                    for j in range(x - 1, x + 2):
                        if not (i == y and j == x) and skeleton[i, j] != 0:
                            num_neighbors += 1
                if num_neighbors == 1:
                    end_points.append(point)
            if not end_points:  # 检查end_points是否为空
                continue  # 如果为空则跳过下面的循环

            # 找到最上部、右侧、左侧和起点的端点坐标
            top_points = []
            bottom_point = end_points[0]
            for point in end_points:
                if len(top_points) < top_count:
                    top_points.append(point)
                elif point[0] < top_points[-1][0]:
                    top_points[-1] = point
                    top_points.sort()
                if point[0] > bottom_point[0]:
                    bottom_point = point

            # 开启则多计算最左边20个端点
            # left_points = []
            # for point in end_points:
            #     if len(left_points) < left_count:
            #         left_points.append(point)
            #     elif point[1] < top_points[-1][1]:
            #         left_points[-1] = point
            #         left_points.sort()
            #
            # top_points.extend(left_points)

            # 执行A*算法找到从起点到终点的最短路径，并计算其中包含的像素点数量
            List_suichang = []
            for point2 in top_points:
                try:
                    path = astar(point2, bottom_point, heuristic)
                    num_pixels = len(path)
                    List_suichang.append(num_pixels)
                except Exception as e:
                    # print(f"发生错误: {e}. 起点不能连接到端点。")
                    continue
            try:
                Max_suichang = max(List_suichang)
            except Exception as e:
                print('该图像处理失败')
                continue
            real_suichang = Max_suichang * pixels_per_mm
            results.append([filename, real_suichang, pixels_per_mm])
            df = pd.DataFrame(results, columns=['文件名', '穗长', '转换比率'])
            df.to_excel(output_file, index=False)

            print(f"稻穗{filename}的穗长为：", real_suichang, 'mm', '转换比率:', {pixels_per_mm})

            # 找到最大值所对应的路径
            max_path_index = List_suichang.index(Max_suichang)
            # max_path = astar(top_points[max_path_index], bottom_point, heuristic)

            max_path = astar(top_points[max_path_index], bottom_point, heuristic)
            if max_path is None:
                print("No path found!")
            else:
                canvas = np.zeros(skeleton.shape)  # 新建画布并画出路径
                for point in max_path:
                    skeleton[point[0] - 3:point[0] + 4, point[1] - 3:point[1] + 4] = 255  # 加粗主路径为8个像素粗度

                result_img = Image.fromarray(skeleton.astype(np.uint8))  # 转换图片格式为unit8
                result_img.save(os.path.join(output_folder, filename))  #保存图片到指定路径


def get_panicle_length_neck(input_folder, Uncut_img_folder, output_folder, output_file, NOP_txt):
    '''
    :param input_folder:  # 裁剪后稻穗图片文件夹
    :param Uncut_img_folder: # 未经过裁剪 带有标定物的原图
    :param output_folder: # 保存稻穗主路径图像的文件夹
    :param output_file: # 存放穗长数据的表格
    :param NOP_txt: # 存放穗颈节点的位置
    :return:
    '''
    def astar(start, goal, h):  # A*算法
        frontier = [(0, start)]
        came_from = {}
        cost_so_far = {start: 0}
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break

            for neighbor in neighbors(current):
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + h(neighbor, goal)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

                    # 重建路径
        path = []
        current = goal
        while current != start:
            path.append(current)
            if current not in came_from:  # 检查came_from字典中是否存在当前键值
                break  # 如果不存在，跳出循环，防止KeyError
            current = came_from[current]
        path.append(start)
        path.reverse()

        return path if path else None  # 如果没有找到路径，返回None

    # 计算启发式估价函数h
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def neighbors(point):  # 获取一个点的所有相邻点
        y, x = point
        result = []
        for i in range(y - 1, y + 2):
            for j in range(x - 1, x + 2):
                if not (i == y and j == x) and skeleton[i, j] != 0:
                    result.append((i, j))
        return result
    for filename in tqdm(os.listdir(input_folder)):  # 遍历文件夹中的所有图片
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # 读取图片
            img = cv2.imread(os.path.join(input_folder, filename))
            filename_uncut = filename.replace('.png', '.jpg')
            img_uncut = cv2.imread(os.path.join(Uncut_img_folder, filename_uncut))
            img_rgb = cv2.cvtColor(img_uncut, cv2.COLOR_BGR2RGB)  # 将图像转换成RGB颜色空间
            # 读取对应图片的txt文件，获取起点的y坐标比例
            with open(os.path.join(NOP_txt, filename.replace('.png', '.txt')), 'r') as file:
                y_coordinate_ratio = float(file.read().split()[2])

            # 获取图像高度，并据此计算起点y坐标
            image_height = img.shape[0]
            start_y = int(image_height * y_coordinate_ratio)
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

                # print(filename, pixels_per_mm)

            # cv2.drawContours(black_image, [max_contour], 0, 255, -1)
            # cv2.imwrite(os.path.join(output_folder, filename), black_image)

            mask_img = cv2.inRange(img, lower, upper)  # 阈值分割
            roi = cv2.bitwise_and(img, img, mask=mask_img)  # 提取感兴趣区域
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # 将感兴趣区域转为为hsv空间
            _, _, s = cv2.split(hsv)  # 提取S分量图
            _, thresh = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津法（Otsu）阈值分割
            # cv2.imwrite(os.path.join(output_folder, filename), thresh)

            # 滤除面积小于阈值的部分
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < area_threshold:
                    cv2.drawContours(thresh, [contour], -1, 0, -1)
            # cv2.imwrite(os.path.join(output_folder, filename), thresh)

            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            for i in range(len(contours)):
                if hierarchy[0][i][3] != -1:  # 检测当前轮廓是否为孔洞
                    hole_area = cv2.contourArea(contours[i])  # 计算孔洞的面积

                    if hole_area < HOLE_AREA_THRESHOLD:  # 如果孔洞的面积小于阈值
                        cv2.drawContours(thresh, [contours[i]], 0, 255, -1)  # 进行像素填充处理
            kernel = np.ones((9, 9), np.uint8)  # 定义结构元素（kernal）的大小和形状，这里我们使用9*9的矩形结构元素
            dilated = cv2.dilate(thresh, kernel, iterations=2)
            # cv2.imwrite(os.path.join(output_folder, filename), dilated)

            # 提取图像骨架并
            skeleton = skeletonize(dilated)
            skeleton = np.uint8(skeleton) * 255
            skeleton = np.pad(skeleton, ((1, 1), (1, 1)), mode='constant', constant_values=0)
            # cv2.imwrite(os.path.join(output_folder, filename), skeleton)

            # 获取骨架中所有点
            points = []
            for S_point in zip(*np.where(skeleton == 255)):
                points.append(S_point)

            # 寻找所有端点
            end_points = []
            for point in points:
                y, x = point
                num_neighbors = 0
                for i in range(y - 1, y + 2):
                    for j in range(x - 1, x + 2):
                        if not (i == y and j == x) and skeleton[i, j] != 0:
                            num_neighbors += 1
                if num_neighbors == 1:
                    end_points.append(point)
            if not end_points:  # 检查end_points是否为空
                continue  # 如果为空则跳过下面的循环

            # 找到最上部、右侧、左侧的端点坐标
            top_points = []
            for point in end_points:
                if len(top_points) < top_count:
                    top_points.append(point)
                elif point[0] < top_points[-1][0]:
                    top_points[-1] = point
                    top_points.sort()

            # # 处理多个起点的情况，取第一个起点为准
            # start_y_coordinates = [point[0] for point in top_points]
            # start_y = min(start_y_coordinates)

            bottom_point = None
            for S_point in zip(*np.where(skeleton == 255)):
                if S_point[0] == start_y:
                    bottom_point = S_point

            # 如果 bottom_point 仍然是 None，则说明没有找到有效的起点和终点，跳过当前图像的处理
            if bottom_point is None:
                print(f'图像{filename}处理失败')
                continue

            # 开启则多计算最左边20个端点
            # left_points = []
            # for point in end_points:
            #     if len(left_points) < left_count:
            #         left_points.append(point)
            #     elif point[1] < top_points[-1][1]:
            #         left_points[-1] = point
            #         left_points.sort()
            #
            # top_points.extend(left_points)

            # 执行A*算法找到从起点到终点的最短路径，并计算其中包含的像素点数量
            List_suichang = []
            for point2 in top_points:
                try:
                    path = astar(point2, bottom_point, heuristic)
                    num_pixels = len(path)
                    List_suichang.append(num_pixels)
                except Exception as e:
                    # print(f"发生错误: {e}. 起点不能连接到端点。")
                    continue
            try:
                Max_suichang = max(List_suichang)
            except Exception as e:
                print('该图像处理失败')
                continue
            real_suichang = Max_suichang * pixels_per_mm
            results.append([filename, real_suichang, pixels_per_mm])
            df = pd.DataFrame(results, columns=['文件名', '穗长', '转换比率'])
            df.to_excel(output_file, index=False)

            print(f"稻穗{filename}的穗长为：", real_suichang, 'mm', '转换比率:', {pixels_per_mm})

            # 找到最大值所对应的路径
            max_path_index = List_suichang.index(Max_suichang)
            # max_path = astar(top_points[max_path_index], bottom_point, heuristic)

            max_path = astar(top_points[max_path_index], bottom_point, heuristic)
            if max_path is None:
                print("No path found!")
            else:
                for point in max_path:
                    skeleton[point[0] - 3:point[0] + 4, point[1] - 3:point[1] + 4] = 255  # 加粗主路径为8个像素粗度

                result_img = Image.fromarray(skeleton.astype(np.uint8))  # 转换图片格式为unit8
                result_img.save(os.path.join(output_folder, filename))  #保存图片到指定路径



if __name__ == '__main__':
    get_panicle_length_neck(input_folder=r"D:\test2024\0117\test_PL_IMG\F_img",
                            Uncut_img_folder=r"D:\test2024\0117\test_PL_IMG\rename",
                            output_folder=r"D:\test2024\0117\test_PL_IMG\mainpath",
                            output_file=r"D:\test2024\0117\0117result.xlsx",
                            NOP_txt=r"D:\test2024\0117\test_PL_IMG\predict11\labels")






