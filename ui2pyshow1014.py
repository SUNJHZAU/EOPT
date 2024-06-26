import heapq,os,sys,cv2
import numpy as np
import pandas as pd
import onnxruntime as ort
from PIL import Image, ImageOps
from PyQt5.QtGui import QPixmap
from skimage.morphology import skeletonize
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem, QMainWindow, QApplication
sys.setrecursionlimit(1000000000)

class YOLOv8:
    """YOLOv8 object detection model class for handling inference and visualization."""

    def __init__(self, onnx_model, confidence_thres, iou_thres):
        """
        Initializes an instance of the YOLOv8 class.

        Args:
            onnx_model: Path to the ONNX model.
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
        self.onnx_model = onnx_model
        # self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # Create an inference session using the ONNX model and specify execution providers
        self.session = ort.InferenceSession(self.onnx_model,
                                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        # Get the model inputs
        self.model_inputs = self.session.get_inputs()

        # Store the shape of the input for later use
        input_shape = self.model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

    def preprocess(self, img_path):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Read the input image using OpenCV
        self.img = cv2.imread(img_path)

        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data

    def postprocess(self, input_image, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """

        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        out = []
        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            out.append({'box': box,
                        'score': score,
                        'clsID': class_id})
            # Draw the detection on the input image
            # self.draw_detections(input_image, box, score, class_id)

        # Return the modified input image
        return out

    def main(self, img_path):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """
        # # Create an inference session using the ONNX model and specify execution providers
        # session = ort.InferenceSession(self.onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        #
        # # Get the model inputs
        # model_inputs = session.get_inputs()
        #
        # # Store the shape of the input for later use
        # input_shape = model_inputs[0].shape
        # self.input_width = input_shape[2]
        # self.input_height = input_shape[3]

        # Preprocess the image data
        img_data = self.preprocess(img_path)

        # Run inference using the preprocessed image data
        outputs = self.session.run(None, {self.model_inputs[0].name: img_data})

        # Perform post-processing on the outputs to obtain output image.
        return self.postprocess(self.img, outputs)  # output image

class Panicle(object):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.detection_LW = YOLOv8("GrainNuber.onnx", 0.7, 0.8)

        self.grandparent_dir = os.path.dirname(self.img_dir)  # 获取图片所在目录的祖父目录
        self.output_saveSubGrain = os.path.join(self.grandparent_dir, 'Subimg_Grain')  # 创建输出目录
        if not os.path.exists(self.output_saveSubGrain):
            os.makedirs(self.output_saveSubGrain)
        self.output_saveSubGrainProcess = os.path.join(self.grandparent_dir, 'Subimg_ProcessGrain')  # 创建输出目录
        if not os.path.exists(self.output_saveSubGrainProcess):
            os.makedirs(self.output_saveSubGrainProcess)

        self.result_all = []

    # def get_kernalLW(self):
    #     results = []
    #     lower_RGB, upper_RGB = (26, 109, 159), (134, 189, 214)  # 定义RGB阈值
    #     # lower_red, upper_red = np.array([138, 27, 32]), np.array([248, 100, 108])  # 定义标定物的rgb颜色范围
    #     lower_red, upper_red = np.array([105, 0, 15]), np.array([255, 100, 255])  # 定义标定物的rgb颜色范围
    #     real_radius_mm = 25  # 已知红色实心圆在现实世界中的大小（单位：mm）
    #     def get_two_letters(j):
    #         # 如果序号小于26，直接返回一个大写字母
    #         if j < 26:
    #             return chr(ord('A') + j)
    #         # 否则，返回两个大写字母，第一个是商，第二个是余数
    #         else:
    #             return chr(ord('A') + j // 26 - 1) + chr(ord('A') + j % 26)
    #
    #     for file in os.listdir(self.output_saveCut):
    #         rectangles = []  # 此图片所有标注框的坐标信息
    #         if file.endswith('.jpg'):
    #             img = Image.open(os.path.join(self.output_saveCut, file))  # 打开裁剪图
    #             img_uncut = cv2.imread(os.path.join(self.output_saveRename, file))
    #             output_LW = self.detection_LW.main(os.path.join(self.output_saveCut, file))
    #
    #             # 进行距离矫正，找到合适的像素对真实世界转换比率
    #             img_rgb = cv2.cvtColor(img_uncut, cv2.COLOR_BGR2RGB)  # 将图像转换成RGB颜色空间
    #             mask_uncut = cv2.inRange(img_rgb, lower_red, upper_red)  # 根据颜色范围进行红色实心圆的掩码
    #             kernal = np.ones((7, 7), np.uint8)  # 定义结构元素（kernal）的大小和形状，这里我们使用7x7的矩形结构元素
    #             img_close = cv2.morphologyEx(mask_uncut, cv2.MORPH_CLOSE, kernal)  # 使用闭运算（close）来消除小物体或者去除物体中的小洞
    #             contours, _ = cv2.findContours(img_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
    #
    #             # 假设只有一个红色实心圆，选择最大的轮廓
    #             pixels_per_mm = None
    #             if len(contours) > 0:
    #                 max_contour = max(contours, key=cv2.contourArea)
    #                 (x, y), radius = cv2.minEnclosingCircle(max_contour)
    #                 radius_in_pixels = radius  # 红色实心圆的半径（像素）
    #                 pixels_per_mm = real_radius_mm / radius_in_pixels  # 计算每个像素对应的实际尺寸（mm）
    #
    #             image_name = os.path.splitext(os.path.basename(os.path.join(self.output_saveCut, file)))[0]  # 去除后缀的图片名
    #             # 遍历output中的每个字典
    #             for item in output_LW:
    #                 box = item['box']  # 获取'box'键对应的列表
    #                 x1, y1, w, h = box  # 假设box列表的格式为[x1, y1, width, height]
    #                 x2 = x1 + w  # 计算x2
    #                 y2 = y1 + h  # 计算y2
    #                 coordinate = (x1, y1, x2, y2)  # 组成坐标元组
    #                 rectangles.append(coordinate)  # 将坐标添加到列表中
    #
    #             # 指定目录，如果不存在就创建
    #             directory = os.path.join(self.output_saveSubGrain, image_name)
    #             if not os.path.exists(directory):
    #                 os.makedirs(directory)
    #
    #             # 遍历矩形列表，复制出子图，保存到文件夹中，按照图片名加上英文序列号命名
    #             for i, rectangle in enumerate(rectangles):
    #                 padded_img = ImageOps.expand(img.crop(rectangle), border=(15, 15, 15, 15), fill=1)  # 复制子图并添加 padding
    #                 padded_img.save(os.path.join(directory, f"{image_name}{get_two_letters(i)}.jpg"))  # 保存子图到文件夹中，拼接子图的完整路径
    #
    #             sub_save_folder = os.path.join(self.output_saveSubGrainProcess, image_name)
    #             if not os.path.exists(sub_save_folder):
    #                 os.makedirs(sub_save_folder)
    #             List_chang = []
    #             List_kuan, List_kuanT5 = [], []
    #             List_ratio = []
    #             List_None = []
    #             sub_folder_path = os.path.join(self.output_saveSubGrain, image_name)
    #
    #             for sub_filename in os.listdir(sub_folder_path):
    #                 sub_path = os.path.join(sub_folder_path, sub_filename)
    #                 if sub_filename.endswith('.jpg'):
    #                     # 读取图像
    #                     img = cv2.imread(os.path.join(sub_folder_path, sub_filename))
    #                     # cv2.imshow('原图', img)
    #                     # cv2.waitKey(25)
    #
    #                     # RGB阈值分割
    #                     mask = cv2.inRange(img, lower_RGB, upper_RGB)
    #                     # cv2.imshow('阈值分割', mask)
    #                     # cv2.waitKey(25)
    #
    #                     # 只保留最大目标
    #                     contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #                     max_area = 0
    #                     max_contour = None
    #                     for contour in contours:
    #                         area = cv2.contourArea(contour)
    #
    #                         if area > max_area:
    #                             max_area = area
    #                             max_contour = contour
    #                     mask.fill(0)
    #                     if max_contour is not None:
    #                         cv2.drawContours(mask, [max_contour], 0, 255, -1)
    #
    #                     # 填充空洞
    #                     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    #                     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #                     # cv2.imshow('空洞填充', mask)
    #                     # cv2.waitKey(25)
    #
    #                     # 计算最小外接矩形并绘制
    #                     try:
    #                         rect = cv2.minAreaRect(max_contour)
    #                     except cv2.error:
    #                         continue
    #                     # rect = cv2.minAreaRect(max_contour)
    #                     # box = cv2.boxPoints(rect)
    #
    #                     # img.save(os.path.join(sub_save_folder, sub_filename))
    #                     w, h = rect[1]
    #                     rate = max(w, h) / min(w, h)
    #                     if 1.7 < rate < 6.4 and 49.92 < max(w, h) < 135.20 and 14.04 < min(w, h) < 42.64:
    #                         List_chang.append(max(w, h))
    #                         List_kuan.append(min(w, h))
    #                         List_ratio.append(max(w, h) / min(w, h))
    #                         # box = box.astype(int)
    #                         # cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
    #                         # pil_img = Image.fromarray(img)
    #                         # cv2.imshow('Final Image', img)
    #                         # cv2.waitKey(25)
    #                         # pil_img.save(os.path.join(sub_save_folder, sub_filename))
    #             if len(List_chang) >= 3:
    #                 # # 遍历List_chang，去除一个最大值和一个最小值
    #                 # List_chang.remove(max(List_chang))
    #                 # List_chang.remove(min(List_chang))
    #                 #
    #                 # # 去除List_kuan和List_ratio对应位置的值
    #                 # List_kuan.pop(List_chang.index(max(List_chang)))
    #                 # List_ratio.pop(List_chang.index(max(List_chang)))
    #                 # List_kuan.pop(List_chang.index(min(List_chang)))
    #                 # List_ratio.pop(List_chang.index(min(List_chang)))
    #                 #
    #                 # # 遍历List_kuan，去除一个最大值和一个最小值
    #                 # List_kuan.remove(max(List_kuan))
    #                 # List_kuan.remove(min(List_kuan))
    #                 #
    #                 # # 去除List_chang和List_ratio对应位置的值
    #                 # List_chang.pop(List_kuan.index(max(List_kuan)))
    #                 # List_ratio.pop(List_kuan.index(max(List_kuan)))
    #                 # List_chang.pop(List_kuan.index(min(List_kuan)))
    #                 # List_ratio.pop(List_kuan.index(min(List_kuan)))
    #                 #
    #                 # # 遍历List_ratio，去除一个最大值和一个最小值
    #                 # List_ratio.remove(max(List_ratio))
    #                 # List_ratio.remove(min(List_ratio))
    #                 #
    #                 # # 去除List_chang和List_kuan对应位置的值
    #                 # List_chang.pop(List_ratio.index(max(List_ratio)))
    #                 # List_kuan.pop(List_ratio.index(max(List_ratio)))
    #                 # List_chang.pop(List_ratio.index(min(List_ratio)))
    #                 # List_kuan.pop(List_ratio.index(min(List_ratio)))
    #                 # List_kuan_max_5 = sorted(List_kuan, reverse=True)[:5]
    #                 # List_ratio_min_5 = sorted(List_ratio)[:5]
    #                 A_chang = sum(List_chang) / len(List_chang)
    #                 A_kuan = sum(List_kuan) / len(List_kuan)
    #                 A_ratio = sum(List_ratio) / len(List_ratio)
    #                 # A_kuant5 = sum(List_kuan_max_5) / 5
    #                 # A_ratiof5 = sum(List_ratio_min_5) / 5
    #                 real_A_chang = A_chang * pixels_per_mm
    #                 real_A_kuan = A_kuan * pixels_per_mm
    #                 # real_A_kuant5 = A_kuant5 * Exange_S
    #                 # results.append([image_name, real_A_chang, real_A_kuan, real_A_kuant5, A_ratio, A_ratiof5])
    #                 results.append([image_name, real_A_chang, real_A_kuan, A_ratio, pixels_per_mm])
    #                 self.result_all.append([image_name, real_A_chang, real_A_kuan])
    #                 # print(f"文件{image_name}已经计算完成  粒长:{real_A_chang}  粒宽：{real_A_kuan}   粒宽t5：{real_A_kuant5}  长宽比:{A_ratio} 长宽比f5：{A_ratiof5}")
    #                 # print(f"{image_name}  粒长:{real_A_chang}  粒宽：{real_A_kuan}   长宽比:{A_ratio} ")
    #             else:
    #                 List_None.append(image_name)
    #     output_file = os.path.join(self.grandparent_dir, 'Result.xlsx')
    #     df = pd.DataFrame(results, columns=['文件名', '粒长', '粒宽', '长宽比', '单像素对应毫米'])
    #     if os.path.isfile(output_file):
    #         # 文件存在，以追加模式写入
    #         with pd.ExcelWriter(output_file, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
    #             df.to_excel(writer, sheet_name='Grain_traits', index=False, header=False)
    #     else:
    #         df.to_excel(output_file, index=False)


class Ui_Form(object):
    def __init__(self):
        self.detection_LW = YOLOv8("GrainNuber.onnx", 0.7, 0.8)

    def setupUi(self, Form):
        Form.setObjectName("Extraction of rice panicle traits")
        Form.resize(979, 522)
        self.horizontalLayoutWidget = QtWidgets.QWidget(Form)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 1001, 31))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.btn_Show_window = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.btn_Show_window.setObjectName("pushButton")
        self.horizontalLayout_2.addWidget(self.btn_Show_window)
        self.btn_PL_window = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.btn_PL_window.setObjectName("btn_PL_window")
        self.horizontalLayout_2.addWidget(self.btn_PL_window)
        self.btn_LW_window = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.btn_LW_window.setObjectName("btn_LW_window")
        self.horizontalLayout_2.addWidget(self.btn_LW_window)
        self.stackedWidget = QtWidgets.QStackedWidget(Form)
        self.stackedWidget.setGeometry(QtCore.QRect(19, 49, 951, 621))
        self.stackedWidget.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.stackedWidget.setFrameShadow(QtWidgets.QFrame.Raised)
        self.stackedWidget.setLineWidth(2)
        self.stackedWidget.setMidLineWidth(2)
        self.stackedWidget.setObjectName("stackedWidget")
        self.page_show = QtWidgets.QWidget()
        self.page_show.setObjectName("page_show")
        self.verticalLayoutWidget_5 = QtWidgets.QWidget(self.page_show)
        self.verticalLayoutWidget_5.setGeometry(QtCore.QRect(150, 0, 112, 371))
        self.verticalLayoutWidget_5.setObjectName("verticalLayoutWidget_5")
        self.show_choose_verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_5)
        self.show_choose_verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.show_choose_verticalLayout.setObjectName("show_choose_verticalLayout")
        self.show_choose_btn_cuti_mg = QtWidgets.QPushButton(self.verticalLayoutWidget_5)
        self.show_choose_btn_cuti_mg.setObjectName("show_choose_btn_cuti_mg")
        self.show_choose_verticalLayout.addWidget(self.show_choose_btn_cuti_mg)
        self.show_choose_btn_raw_img = QtWidgets.QPushButton(self.verticalLayoutWidget_5)
        self.show_choose_btn_raw_img.setObjectName("show_choose_btn_raw_img")
        self.show_choose_verticalLayout.addWidget(self.show_choose_btn_raw_img)
        self.show_choose_btn_txt = QtWidgets.QPushButton(self.verticalLayoutWidget_5)
        self.show_choose_btn_txt.setObjectName("show_choose_btn_txt")
        self.show_choose_verticalLayout.addWidget(self.show_choose_btn_txt)
        self.show_choose_btn_sub_img = QtWidgets.QPushButton(self.verticalLayoutWidget_5)
        self.show_choose_btn_sub_img.setObjectName("show_choose_btn_sub_img")
        self.show_choose_verticalLayout.addWidget(self.show_choose_btn_sub_img)
        self.show_choose_btn_sub_img_process = QtWidgets.QPushButton(self.verticalLayoutWidget_5)
        self.show_choose_btn_sub_img_process.setObjectName("show_choose_btn_sub_img_process")
        self.show_choose_verticalLayout.addWidget(self.show_choose_btn_sub_img_process)
        self.show_label_window_raw_img = QtWidgets.QLabel(self.page_show)
        self.show_label_window_raw_img.setGeometry(QtCore.QRect(270, 60, 211, 291))
        self.show_label_window_raw_img.setFrameShape(QtWidgets.QFrame.Panel)
        self.show_label_window_raw_img.setText("")
        self.show_label_window_raw_img.setObjectName("show_label_window_raw_img")
        self.show_btn_start = QtWidgets.QPushButton(self.page_show)
        self.show_btn_start.setGeometry(QtCore.QRect(0, 380, 249, 51))
        self.show_btn_start.setAutoDefault(True)
        self.show_btn_start.setDefault(True)
        self.show_btn_start.setFlat(False)
        self.show_btn_start.setObjectName("show_btn_start")
        self.show_label_window_panicle = QtWidgets.QLabel(self.page_show)
        self.show_label_window_panicle.setGeometry(QtCore.QRect(500, 60, 211, 291))
        self.show_label_window_panicle.setFrameShape(QtWidgets.QFrame.Panel)
        self.show_label_window_panicle.setText("")
        self.show_label_window_panicle.setObjectName("show_label_window_panicle")
        self.show_label_window_grain = QtWidgets.QLabel(self.page_show)
        self.show_label_window_grain.setGeometry(QtCore.QRect(730, 60, 211, 291))
        self.show_label_window_grain.setFrameShape(QtWidgets.QFrame.Panel)
        self.show_label_window_grain.setText("")
        self.show_label_window_grain.setObjectName("show_label_window_grain")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.page_show)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(270, 10, 671, 51))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.show_horizontalLayout_title = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.show_horizontalLayout_title.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.show_horizontalLayout_title.setContentsMargins(0, 0, 0, 0)
        self.show_horizontalLayout_title.setObjectName("show_horizontalLayout_title")
        self.show_label_title_raw_img = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.show_label_title_raw_img.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.show_label_title_raw_img.setAlignment(QtCore.Qt.AlignCenter)
        self.show_label_title_raw_img.setObjectName("show_label_title_raw_img")
        self.show_horizontalLayout_title.addWidget(self.show_label_title_raw_img)
        self.show_label_title_panicle = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.show_label_title_panicle.setAlignment(QtCore.Qt.AlignCenter)
        self.show_label_title_panicle.setObjectName("show_label_title_panicle")
        self.show_horizontalLayout_title.addWidget(self.show_label_title_panicle)
        self.show_label_title_Grain = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.show_label_title_Grain.setAlignment(QtCore.Qt.AlignCenter)
        self.show_label_title_Grain.setObjectName("show_label_title_Grain")
        self.show_horizontalLayout_title.addWidget(self.show_label_title_Grain)
        self.formLayoutWidget_4 = QtWidgets.QWidget(self.page_show)
        self.formLayoutWidget_4.setGeometry(QtCore.QRect(0, 0, 151, 528))
        self.formLayoutWidget_4.setObjectName("formLayoutWidget_4")
        self.Show_formLayout = QtWidgets.QFormLayout(self.formLayoutWidget_4)
        self.Show_formLayout.setSizeConstraint(QtWidgets.QLayout.SetMinAndMaxSize)
        self.Show_formLayout.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)
        self.Show_formLayout.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)
        self.Show_formLayout.setContentsMargins(6, 8, 7, 0)
        self.Show_formLayout.setHorizontalSpacing(7)
        self.Show_formLayout.setVerticalSpacing(18)
        self.Show_formLayout.setObjectName("Show_formLayout")
        self.show_label_cut_img = QtWidgets.QLabel(self.formLayoutWidget_4)
        self.show_label_cut_img.setObjectName("show_label_cut_img")
        self.Show_formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.show_label_cut_img)
        self.show_Edit_cut_img = QtWidgets.QLineEdit(self.formLayoutWidget_4)
        self.show_Edit_cut_img.setObjectName("show_Edit_cut_img")
        self.Show_formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.show_Edit_cut_img)
        self.show_label_raw_img = QtWidgets.QLabel(self.formLayoutWidget_4)
        self.show_label_raw_img.setObjectName("show_label_raw_img")
        self.Show_formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.show_label_raw_img)
        self.show_Edit_raw_img = QtWidgets.QLineEdit(self.formLayoutWidget_4)
        self.show_Edit_raw_img.setObjectName("show_Edit_raw_img")
        self.Show_formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.show_Edit_raw_img)
        self.show_label_txt = QtWidgets.QLabel(self.formLayoutWidget_4)
        self.show_label_txt.setObjectName("show_label_txt")
        self.Show_formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.show_label_txt)
        self.show_Edit_txt = QtWidgets.QLineEdit(self.formLayoutWidget_4)
        self.show_Edit_txt.setObjectName("show_Edit_txt")
        self.Show_formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.show_Edit_txt)
        self.show_label_sub_img = QtWidgets.QLabel(self.formLayoutWidget_4)
        self.show_label_sub_img.setObjectName("show_label_sub_img")
        self.Show_formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.show_label_sub_img)
        self.show_Edit_sub_img = QtWidgets.QLineEdit(self.formLayoutWidget_4)
        self.show_Edit_sub_img.setObjectName("show_Edit_sub_img")
        self.Show_formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.show_Edit_sub_img)
        self.show_label_sub_img_process = QtWidgets.QLabel(self.formLayoutWidget_4)
        self.show_label_sub_img_process.setEnabled(True)
        self.show_label_sub_img_process.setObjectName("show_label_sub_img_process")
        self.Show_formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.show_label_sub_img_process)
        self.show_Edit_sub_img_process = QtWidgets.QLineEdit(self.formLayoutWidget_4)
        self.show_Edit_sub_img_process.setObjectName("show_Edit_sub_img_process")
        self.Show_formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.show_Edit_sub_img_process)
        self.show_textedit_showdata = QtWidgets.QTableWidget(self.page_show)
        self.show_textedit_showdata.setGeometry(QtCore.QRect(270, 360, 671, 81))
        self.show_textedit_showdata.setColumnCount(6)
        self.show_textedit_showdata.setObjectName("show_TableWidget_showdata")

        headers = ["Image Name", "Panicle lengrh", "Grain length", "Grain width", "Aspect ratio", "Pixels per mm"]
        self.show_textedit_showdata.setHorizontalHeaderLabels(headers)

        self.stackedWidget.addWidget(self.page_show)
        self.page_LW = QtWidgets.QWidget()
        self.page_LW.setObjectName("page_LW")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.page_LW)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(30, 320, 251, 91))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.LW_verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.LW_verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.LW_verticalLayout.setObjectName("LW_verticalLayout")
        self.LW_btn_start = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.LW_btn_start.setAutoDefault(True)
        self.LW_btn_start.setDefault(True)
        self.LW_btn_start.setFlat(False)
        self.LW_btn_start.setObjectName("LW_btn_start")
        self.LW_verticalLayout.addWidget(self.LW_btn_start)
        self.LW_btn_stop = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.LW_btn_stop.setAutoDefault(True)
        self.LW_btn_stop.setDefault(True)
        self.LW_btn_stop.setFlat(False)
        self.LW_btn_stop.setObjectName("LW_btn_stop")
        self.LW_verticalLayout.addWidget(self.LW_btn_stop)
        self.LW_scrollArea = QtWidgets.QScrollArea(self.page_LW)
        self.LW_scrollArea.setGeometry(QtCore.QRect(380, 30, 531, 381))
        self.LW_scrollArea.setWidgetResizable(True)
        self.LW_scrollArea.setObjectName("LW_scrollArea")
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 529, 379))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.LW_scrollArea.setWidget(self.scrollAreaWidgetContents_2)
        self.formLayoutWidget_3 = QtWidgets.QWidget(self.page_LW)
        self.formLayoutWidget_3.setGeometry(QtCore.QRect(30, 30, 251, 281))
        self.formLayoutWidget_3.setObjectName("formLayoutWidget_3")
        self.LW_formLayout = QtWidgets.QFormLayout(self.formLayoutWidget_3)
        self.LW_formLayout.setSizeConstraint(QtWidgets.QLayout.SetMinAndMaxSize)
        self.LW_formLayout.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)
        self.LW_formLayout.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)
        self.LW_formLayout.setContentsMargins(5, 9, 6, 9)
        self.LW_formLayout.setObjectName("LW_formLayout")
        self.LW_label_cut_img = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.LW_label_cut_img.setObjectName("LW_label_cut_img")
        self.LW_formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.LW_label_cut_img)
        self.LW_Edit_cut_img = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.LW_Edit_cut_img.setObjectName("LW_Edit_cut_img")
        self.LW_formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.LW_Edit_cut_img)
        self.LW_label_raw_img = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.LW_label_raw_img.setObjectName("LW_label_raw_img")
        self.LW_formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.LW_label_raw_img)
        self.LW_Edit_raw_img = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.LW_Edit_raw_img.setObjectName("LW_Edit_raw_img")
        self.LW_formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.LW_Edit_raw_img)
        self.LW_label_txt = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.LW_label_txt.setObjectName("LW_label_txt")
        self.LW_formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.LW_label_txt)
        self.LW_Edit_txt = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.LW_Edit_txt.setObjectName("LW_Edit_txt")
        self.LW_formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.LW_Edit_txt)
        self.LW_label_sub_img = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.LW_label_sub_img.setObjectName("LW_label_sub_img")
        self.LW_formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.LW_label_sub_img)
        self.LW_Edit_sub_img = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.LW_Edit_sub_img.setObjectName("LW_Edit_sub_img")
        self.LW_formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.LW_Edit_sub_img)
        self.LW_label_sub_img_process = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.LW_label_sub_img_process.setEnabled(True)
        self.LW_label_sub_img_process.setObjectName("LW_label_sub_img_process")
        self.LW_formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.LW_label_sub_img_process)
        self.LW_Edit_sub_img_process = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.LW_Edit_sub_img_process.setObjectName("LW_Edit_sub_img_process")
        self.LW_formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.LW_Edit_sub_img_process)
        self.LW_label_output = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.LW_label_output.setObjectName("LW_label_output")
        self.LW_formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.LW_label_output)
        self.LW_Edit_output = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.LW_Edit_output.setObjectName("LW_Edit_output")
        self.LW_formLayout.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.LW_Edit_output)
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.page_LW)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(280, 40, 91, 281))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.LW_choose_verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.LW_choose_verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.LW_choose_verticalLayout.setObjectName("LW_choose_verticalLayout")
        self.LW_choose_btn = QtWidgets.QPushButton(self.verticalLayoutWidget_3)
        self.LW_choose_btn.setObjectName("LW_choose_btn")
        self.LW_choose_verticalLayout.addWidget(self.LW_choose_btn)
        self.LW_choose_btn_2 = QtWidgets.QPushButton(self.verticalLayoutWidget_3)
        self.LW_choose_btn_2.setObjectName("LW_choose_btn_2")
        self.LW_choose_verticalLayout.addWidget(self.LW_choose_btn_2)
        self.LW_choose_btn_3 = QtWidgets.QPushButton(self.verticalLayoutWidget_3)
        self.LW_choose_btn_3.setObjectName("LW_choose_btn_3")
        self.LW_choose_verticalLayout.addWidget(self.LW_choose_btn_3)
        self.LW_choose_btn_4 = QtWidgets.QPushButton(self.verticalLayoutWidget_3)
        self.LW_choose_btn_4.setObjectName("LW_choose_btn_4")
        self.LW_choose_verticalLayout.addWidget(self.LW_choose_btn_4)
        self.LW_choose_btn_5 = QtWidgets.QPushButton(self.verticalLayoutWidget_3)
        self.LW_choose_btn_5.setObjectName("LW_choose_btn_5")
        self.LW_choose_verticalLayout.addWidget(self.LW_choose_btn_5)
        self.LW_choose_btn_6 = QtWidgets.QPushButton(self.verticalLayoutWidget_3)
        self.LW_choose_btn_6.setObjectName("LW_choose_btn_6")
        self.LW_choose_verticalLayout.addWidget(self.LW_choose_btn_6)
        self.stackedWidget.addWidget(self.page_LW)
        self.page_PL = QtWidgets.QWidget()
        self.page_PL.setObjectName("page_PL")
        self.PL_scrollArea = QtWidgets.QScrollArea(self.page_PL)
        self.PL_scrollArea.setGeometry(QtCore.QRect(380, 40, 521, 291))
        self.PL_scrollArea.setWidgetResizable(True)
        self.PL_scrollArea.setObjectName("PL_scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 519, 289))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.PL_scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.formLayoutWidget = QtWidgets.QWidget(self.page_PL)
        self.formLayoutWidget.setGeometry(QtCore.QRect(20, 40, 251, 191))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.PLformLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.PLformLayout.setSizeConstraint(QtWidgets.QLayout.SetMinAndMaxSize)
        self.PLformLayout.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)
        self.PLformLayout.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)
        self.PLformLayout.setContentsMargins(5, 9, 6, 0)
        self.PLformLayout.setObjectName("PLformLayout")
        self.PL_label_cut_img = QtWidgets.QLabel(self.formLayoutWidget)
        self.PL_label_cut_img.setObjectName("PL_label_cut_img")
        self.PLformLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.PL_label_cut_img)
        self.PL_label_raw_img = QtWidgets.QLabel(self.formLayoutWidget)
        self.PL_label_raw_img.setObjectName("PL_label_raw_img")
        self.PLformLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.PL_label_raw_img)
        self.label_mainpath = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_mainpath.setObjectName("label_mainpath")
        self.PLformLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_mainpath)
        self.label_output = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_output.setObjectName("label_output")
        self.PLformLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_output)
        self.PL_Edit_cut_img = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.PL_Edit_cut_img.setObjectName("PL_Edit_cut_img")
        self.PLformLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.PL_Edit_cut_img)
        self.PL_Edit_raw_img = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.PL_Edit_raw_img.setObjectName("PL_Edit_raw_img")
        self.PLformLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.PL_Edit_raw_img)
        self.PL_Edit_mainpath = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.PL_Edit_mainpath.setObjectName("PL_Edit_mainpath")
        self.PLformLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.PL_Edit_mainpath)
        self.PL_Edit_output = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.PL_Edit_output.setObjectName("PL_Edit_output")
        self.PLformLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.PL_Edit_output)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.page_PL)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(20, 240, 251, 91))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.PL_verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.PL_verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.PL_verticalLayout.setObjectName("PL_verticalLayout")
        self.PL_btn_start = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.PL_btn_start.setAutoDefault(True)
        self.PL_btn_start.setDefault(True)
        self.PL_btn_start.setFlat(False)
        self.PL_btn_start.setObjectName("PL_btn_start")
        self.PL_verticalLayout.addWidget(self.PL_btn_start)
        self.PL_btn_stop = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.PL_btn_stop.setAutoDefault(True)
        self.PL_btn_stop.setDefault(True)
        self.PL_btn_stop.setFlat(False)
        self.PL_btn_stop.setObjectName("PL_btn_stop")
        self.PL_verticalLayout.addWidget(self.PL_btn_stop)
        self.verticalLayoutWidget_4 = QtWidgets.QWidget(self.page_PL)
        self.verticalLayoutWidget_4.setGeometry(QtCore.QRect(270, 50, 91, 191))
        self.verticalLayoutWidget_4.setObjectName("verticalLayoutWidget_4")
        self.PL_choose_verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_4)
        self.PL_choose_verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.PL_choose_verticalLayout.setObjectName("PL_choose_verticalLayout")
        self.PL_choose_btn = QtWidgets.QPushButton(self.verticalLayoutWidget_4)
        self.PL_choose_btn.setObjectName("PL_choose_btn")
        self.PL_choose_verticalLayout.addWidget(self.PL_choose_btn)
        self.PL_choose_btn_2 = QtWidgets.QPushButton(self.verticalLayoutWidget_4)
        self.PL_choose_btn_2.setObjectName("PL_choose_btn_2")
        self.PL_choose_verticalLayout.addWidget(self.PL_choose_btn_2)
        self.PL_choose_btn_3 = QtWidgets.QPushButton(self.verticalLayoutWidget_4)
        self.PL_choose_btn_3.setObjectName("PL_choose_btn_3")
        self.PL_choose_verticalLayout.addWidget(self.PL_choose_btn_3)
        self.PL_choose_btn_4 = QtWidgets.QPushButton(self.verticalLayoutWidget_4)
        self.PL_choose_btn_4.setObjectName("PL_choose_btn_4")
        self.PL_choose_verticalLayout.addWidget(self.PL_choose_btn_4)
        self.stackedWidget.addWidget(self.page_PL)

        self.retranslateUi(Form)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

        # 三个页面的切换槽函数与按钮进行绑定
        self.btn_LW_window.clicked.connect(self.show_page_LW)  # 连接到 show_page_LW 槽函数
        self.btn_PL_window.clicked.connect(self.show_page_PL)  # 连接到 show_page_PL 槽函数
        self.btn_Show_window.clicked.connect(self.show_page_Show)  # 连接到 show_page_Show 槽函数

        # 三个界面的选择文件按钮与输入框进行编制
        self.PL_choose_buttons = []
        self.PL_choose_buttons.extend([self.PL_choose_btn, self.PL_choose_btn_2, self.PL_choose_btn_3, self.PL_choose_btn_4])
        self.PL_choose_line_edits = []
        self.PL_choose_line_edits.extend([self.PL_Edit_cut_img, self.PL_Edit_raw_img, self.PL_Edit_mainpath, self.PL_Edit_output])
        self.LW_choose_buttons = []
        self.LW_choose_buttons.extend([self.LW_choose_btn, self.LW_choose_btn_2, self.LW_choose_btn_3, self.LW_choose_btn_4, self.LW_choose_btn_5,self.LW_choose_btn_6])
        self.LW_choose_line_edits = []
        self.LW_choose_line_edits.extend([self.LW_Edit_cut_img, self.LW_Edit_raw_img, self.LW_Edit_txt, self.LW_Edit_sub_img,self.LW_Edit_sub_img_process, self.LW_Edit_output])
        self.Show_choose_buttons_file = []
        self.Show_choose_buttons_file.extend([self.show_choose_btn_cuti_mg, self.show_choose_btn_raw_img, self.show_choose_btn_txt])
        self.Show_choose_line_edits_file = []
        self.Show_choose_line_edits_file.extend([self.show_Edit_cut_img, self.show_Edit_raw_img, self.show_Edit_txt])
        self.Show_choose_buttons_folder = []
        self.Show_choose_buttons_folder.extend([self.show_choose_btn_sub_img, self.show_choose_btn_sub_img_process])
        self.Show_choose_line_edits_folder = []
        self.Show_choose_line_edits_folder.extend([self.show_Edit_sub_img,self.show_Edit_sub_img_process])

        # 三个界面的选择文件槽函数与按钮进行连接
        for PL_i, PL_button in enumerate(self.PL_choose_buttons):
            PL_button.clicked.connect(lambda checked, index=PL_i: self.PL_choose_folder(index))
        for LW_i, LW_button in enumerate(self.LW_choose_buttons):
            LW_button.clicked.connect(lambda checked, index=LW_i: self.LW_choose_folder(index))
        for Show_i, Show_button in enumerate(self.Show_choose_buttons_file):
            Show_button.clicked.connect(lambda checked, index=Show_i: self.Show_choose_img(index))
        for Showf_i, Showf_button in enumerate(self.Show_choose_buttons_folder):
            Showf_button.clicked.connect(lambda checked, index=Showf_i: self.Show_choose_folder(index))



        # 穗长数据展示框
        self.PL_output_textedit = QtWidgets.QTextEdit(self.page_PL)
        self.PL_output_textedit.setGeometry(QtCore.QRect(380, 40, 521, 291))
        self.PL_output_textedit.setReadOnly(True)  # 使文本框只读
        self.PL_scrollArea.setWidget(self.PL_output_textedit)
        # 籽粒性状数据展示框
        self.LW_output_textedit = QtWidgets.QTextEdit(self.page_LW)
        self.LW_output_textedit.setGeometry(QtCore.QRect(380, 40, 521, 291))
        self.LW_output_textedit.setReadOnly(True)  # 使文本框只读
        self.LW_scrollArea.setWidget(self.LW_output_textedit)

        self.PL_btn_start.clicked.connect(self.PL)  # 连接到PL槽函数
        self.LW_btn_start.clicked.connect(self.LW)  # 连接到LW槽函数
        self.show_btn_start.clicked.connect(self.ALL)  # 连接到ALL槽函数

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.btn_Show_window.setText(_translate("Form", "Analysis of single panicle traits"))
        self.btn_PL_window.setText(_translate("Form", "Panicle length"))
        self.btn_LW_window.setText(_translate("Form", "Grain length、Grain"))
        self.show_choose_btn_cuti_mg.setText(_translate("Form", "Select the file"))
        self.show_choose_btn_raw_img.setText(_translate("Form", "Select the file"))
        self.show_choose_btn_txt.setText(_translate("Form", "Select the file"))
        self.show_choose_btn_sub_img.setText(_translate("Form", "Select the folder"))
        self.show_choose_btn_sub_img_process.setText(_translate("Form", "Select the folder"))
        self.show_btn_start.setText(_translate("Form", "Start analysis"))
        self.show_label_title_raw_img.setText(_translate("Form", "Original image"))
        self.show_label_title_panicle.setText(_translate("Form", "Panicle length"))
        self.show_label_title_Grain.setText(_translate("Form", "Grain traits"))
        self.show_label_cut_img.setText(_translate("Form", "Post-cut pictures"))
        self.show_label_raw_img.setText(_translate("Form", "original images"))
        self.show_label_txt.setText(_translate("Form", "inference result(.txt)"))
        self.show_label_sub_img.setText(_translate("Form", "subgraphs"))
        self.show_label_sub_img_process.setText(_translate("Form", "The processed subgraphs"))
        self.LW_btn_start.setText(_translate("Form", "Start analysis"))
        self.LW_btn_stop.setText(_translate("Form", "Stop"))
        self.LW_label_cut_img.setText(_translate("Form", "Post-cut pictures"))
        self.LW_label_raw_img.setText(_translate("Form", "original images"))
        self.LW_label_txt.setText(_translate("Form", "inference result(.txt)"))
        self.LW_label_sub_img.setText(_translate("Form", "subgraphs"))
        self.LW_label_sub_img_process.setText(_translate("Form", "The processed subgraphs"))
        self.LW_label_output.setText(_translate("Form", "Output EXCEL path"))
        self.LW_choose_btn.setText(_translate("Form", "Select the folder"))
        self.LW_choose_btn_2.setText(_translate("Form", "Select the folder"))
        self.LW_choose_btn_3.setText(_translate("Form", "Select the folder"))
        self.LW_choose_btn_4.setText(_translate("Form", "Select the folder"))
        self.LW_choose_btn_5.setText(_translate("Form", "Select the folder"))
        self.LW_choose_btn_6.setText(_translate("Form", "Select the folder"))
        self.PL_label_cut_img.setText(_translate("Form", "Post-cut pictures"))
        self.PL_label_raw_img.setText(_translate("Form", "original images"))
        self.label_mainpath.setText(_translate("Form", "Rice panicle skeleton image"))
        self.label_output.setText(_translate("Form", "Output EXCEL path"))
        self.PL_btn_start.setText(_translate("Form", "Start analysis"))
        self.PL_btn_stop.setText(_translate("Form", "Stop"))
        self.PL_choose_btn.setText(_translate("Form", "Select the folder"))
        self.PL_choose_btn_2.setText(_translate("Form", "Select the folder"))
        self.PL_choose_btn_3.setText(_translate("Form", "Select the folder"))
        self.PL_choose_btn_4.setText(_translate("Form", "Select the folder"))

    # 槽函数：显示页面 page_LW
    def show_page_LW(self):
        self.stackedWidget.setCurrentIndex(1)  # 切换到 page_LW

    # 槽函数：显示页面 page_PL
    def show_page_PL(self):
        self.stackedWidget.setCurrentIndex(2)  # 切换到 page_PL

    # 槽函数：显示页面 page_Show
    def show_page_Show(self):
        self.stackedWidget.setCurrentIndex(0)  # 切换到 page_Show

    def PL_choose_folder(self, index):
        # 创建并显示文件夹选择对话框
        folder_dialog = QFileDialog()
        folder_dialog.setFileMode(QFileDialog.DirectoryOnly)
        folder_dialog.setOption(QFileDialog.ShowDirsOnly)
        if folder_dialog.exec_() == QFileDialog.Accepted:
            # 获取选择的文件夹路径
            folder_path = folder_dialog.selectedFiles()[0]
            # 将路径填入对应的QLineEdit中
            self.PL_choose_line_edits[index].setText(folder_path)

    # 槽函数：选择文件夹路径与填充路径到输入框的逻辑（粒长、粒宽面板）
    def LW_choose_folder(self, index):
        # 创建并显示文件夹选择对话框
        folder_dialog = QFileDialog()
        folder_dialog.setFileMode(QFileDialog.DirectoryOnly)
        folder_dialog.setOption(QFileDialog.ShowDirsOnly)
        if folder_dialog.exec_() == QFileDialog.Accepted:
            # 获取选择的文件夹路径
            folder_path = folder_dialog.selectedFiles()[0]
            # 将路径填入对应的QLineEdit中
            self.LW_choose_line_edits[index].setText(folder_path)

    def Show_choose_img(self, index):
        # 创建并显示文件夹选择对话框
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.txt)")
        if file_dialog.exec_() == QFileDialog.Accepted:
            # 获取选择的文件夹路径
            folder_path = file_dialog.selectedFiles()[0]
            # 将路径填入对应的QLineEdit中
            self.Show_choose_line_edits_file[index].setText(folder_path)

    def Show_choose_folder(self, index):
        # 创建并显示文件夹选择对话框
        folder_dialog = QFileDialog()
        folder_dialog.setFileMode(QFileDialog.DirectoryOnly)
        folder_dialog.setOption(QFileDialog.ShowDirsOnly)
        if folder_dialog.exec_() == QFileDialog.Accepted:
            # 获取选择的文件夹路径
            folder_path = folder_dialog.selectedFiles()[0]
            # 将路径填入对应的QLineEdit中
            self.Show_choose_line_edits_folder[index].setText(folder_path)

    # 槽函数：调用get_panicle_length功能函数
    def PL(self):
        self.get_panicle_length(input_folder=self.PL_Edit_cut_img.text(),  # 裁剪后稻穗图片文件夹
                                Uncut_img_folder=self.PL_Edit_raw_img.text(),  # 未经过裁剪 带有标定物的原图
                                output_folder=self.PL_Edit_mainpath.text(),  # 保存稻穗主路径图像的文件夹
                                output_file=self.PL_Edit_output.text()  # 存放穗长数据的表格
                                )

    def LW(self):
        self.get_kernalLW(A_folder=self.LW_Edit_cut_img.text(),  # 裁剪图文件夹
                          Uncut_img_folder=self.LW_Edit_raw_img.text(),  # 带有标定物的图片文件夹
                          A=self.LW_Edit_sub_img.text(),  # 分割后每张图片的子图保存路径文件夹
                          B=self.LW_Edit_sub_img_process.text(),  # 保存处理后子文件夹的文件夹
                          output_file=self.LW_Edit_output.text()  # 穗长数据保存的文件
                          )

    def ALL(self):
        self.get_all_traits(cut_img=self.show_Edit_cut_img.text(),
                            raw_img=self.show_Edit_raw_img.text(),
                            sub_folder=self.show_Edit_sub_img.text(),
                            subfolder_process=self.show_Edit_sub_img_process.text()
                            )

    # 功能函数：提取稻穗穗长的具体逻辑
    def get_panicle_length(self, input_folder, Uncut_img_folder, output_folder, output_file='./data.xlsx'):
        lower_red, upper_red = np.array([105, 0, 15]), np.array([255, 100, 255])  # 定义标定物的rgb颜色范围
        lower, upper = (0, 20, 20), (255, 255, 255)  # 指定稻穗的bgr颜色范围
        HOLE_AREA_THRESHOLD = 1000  # 填补空洞阈值
        real_radius_mm = 25  # 已知红色实心圆在现实世界中的大小（单位：mm）
        area_threshold = 37000  # 固定面积阈值
        top_count, left_count, right_count = 30, 20, 20  # 计算顶部，左侧，右侧的端点个数
        results = []  # 存放穗长数据的列表

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

        for filename in os.listdir(input_folder):  # 遍历文件夹中的所有图片
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

                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                for i in range(len(contours)):
                    if hierarchy[0][i][3] != -1:  # 检测当前轮廓是否为孔洞
                        hole_area = cv2.contourArea(contours[i])  # 计算孔洞的面积

                        if hole_area < HOLE_AREA_THRESHOLD:  # 如果孔洞的面积小于阈值
                            cv2.drawContours(thresh, [contours[i]], 0, 255, -1)  # 进行像素填充处理
                kernel = np.ones((9, 9), np.uint8)  # 定义结构元素（kernal）的大小和形状，这里我们使用9*9的矩形结构元素
                dilated = cv2.dilate(thresh, kernel, iterations=2)

                # 提取图像骨架
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

                i = f"稻穗{filename}的穗长为：", real_suichang, 'mm', '转换比率:', {pixels_per_mm}
                # print(f"稻穗{filename}的穗长为：", real_suichang, 'mm', '转换比率:', {pixels_per_mm})
                self.PL_output_textedit.append(str(i))

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
                    result_img.save(os.path.join(output_folder, filename))  # 保存图片到指定路径

    # 功能函数：提取籽粒性状的具体逻辑
    def get_kernalLW(self, A_folder, Uncut_img_folder, A, B, output_file='./data.xlsx'):
        # 定义一个函数，根据序号生成两位字母
        def get_two_letters(j):
            # 如果序号小于26，直接返回一个大写字母
            if j < 26:
                return chr(ord('A') + j)
            # 否则，返回两个大写字母，第一个是商，第二个是余数
            else:
                return chr(ord('A') + j // 26 - 1) + chr(ord('A') + j % 26)
        lower_RGB, upper_RGB = (26, 109, 159), (134, 189, 214)  # 定义RGB阈值
        results = []  # 存放穗长数据的列表
        lower_red, upper_red = np.array([105, 0, 15]), np.array([255, 100, 255])  # 定义标定物的rgb颜色范围
        real_radius_mm = 25  # 已知红色实心圆在现实世界中的大小（单位：mm）

        for file in os.listdir(A_folder):
            rectangles = []  # 此图片所有标注框的坐标信息
            if file.endswith('.jpg'):
                img = cv2.imread(os.path.join(A_folder, file))  # 打开原图
                img_cc = Image.open(os.path.join(A_folder, file))  # 打开原图
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

                output_LW = self.detection_LW.main(os.path.join(A_folder, file))
                # 遍历output中的每个字典
                for item in output_LW:
                    box = item['box']  # 获取'box'键对应的列表
                    x1, y1, w, h = box  # 假设box列表的格式为[x1, y1, width, height]
                    x2 = x1 + w  # 计算x2
                    y2 = y1 + h  # 计算y2
                    coordinate = (x1, y1, x2, y2)  # 组成坐标元组
                    rectangles.append(coordinate)  # 将坐标添加到列表中

                # 指定目录，如果不存在就创建
                directory = os.path.join(A, image_name)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # 遍历矩形列表，复制出子图，保存到文件夹中，按照图片名加上英文序列号命名
                for i, rectangle in enumerate(rectangles):
                    padded_img = ImageOps.expand(img_cc.crop(rectangle), border=(15, 15, 15, 15),
                                                 fill=1)  # 复制子图并添加 padding
                    padded_img.save(
                        os.path.join(directory, f"{image_name}{get_two_letters(i)}.jpg"))  # 保存子图到文件夹中，拼接子图的完整路径



                sub_save_folder = os.path.join(B, image_name)
                if not os.path.exists(sub_save_folder):
                    os.makedirs(sub_save_folder)
                List_chang = []
                List_kuan, List_kuanT5 = [], []
                List_ratio = []
                List_None = []
                sub_folder_path = os.path.join(A, image_name)

                for sub_filename in os.listdir(sub_folder_path):

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
                if len(List_chang) >= 7:
                    A_chang = sum(List_chang) / len(List_chang)
                    A_kuan = sum(List_kuan) / len(List_kuan)
                    A_ratio = sum(List_ratio) / len(List_ratio)
                    real_A_chang = A_chang * pixels_per_mm
                    real_A_kuan = A_kuan * pixels_per_mm
                    results.append([image_name, real_A_chang, real_A_kuan, A_ratio, pixels_per_mm])
                    # print(f"文件{image_name}已经计算完成  粒长:{real_A_chang}  粒宽：{real_A_kuan}   粒宽t5：{real_A_kuant5}  长宽比:{A_ratio} 长宽比f5：{A_ratiof5}")
                    i = f"Panicle{image_name}     GL:{real_A_chang}  GW：{real_A_kuan}   AR:{A_ratio}   PD：{pixels_per_mm}"
                    # print(f"文件{image_name}已经计算完成  粒长:{real_A_chang}  粒宽：{real_A_kuan}   长宽比:{A_ratio}   像素转换比率：{pixels_per_mm}")
                    self.LW_output_textedit.append(str(i))
                else:
                    List_None.append(image_name)

        df = pd.DataFrame(results, columns=['PanicleID', 'GL', 'GW', 'AR', 'PD'])
        df.to_excel(output_file, index=False)

    # 功能函数：展示结果与图片的具体逻辑
    def get_all_traits(self, cut_img, raw_img, sub_folder, subfolder_process):
        lower_red, upper_red = np.array([105, 0, 15]), np.array([255, 100, 255])  # 定义标定物的rgb颜色范围
        lower, upper = (0, 20, 20), (255, 255, 255)  # 指定稻穗的bgr颜色范围
        HOLE_AREA_THRESHOLD = 1000  # 填补空洞阈值
        real_radius_mm = 25  # 已知红色实心圆在现实世界中的大小（单位：mm）
        area_threshold = 37000  # 固定面积阈值
        top_count, left_count, right_count = 30, 20, 20  # 计算顶部，左侧，右侧的端点个数

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

        if cut_img.endswith('.jpg') or cut_img.endswith('.png'):
            # 读取图片
            img = cv2.imread(cut_img)
            img_cc = Image.open(cut_img)
            img_uncut = cv2.imread(raw_img)
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

            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            for i in range(len(contours)):
                if hierarchy[0][i][3] != -1:  # 检测当前轮廓是否为孔洞
                    hole_area = cv2.contourArea(contours[i])  # 计算孔洞的面积

                    if hole_area < HOLE_AREA_THRESHOLD:  # 如果孔洞的面积小于阈值
                        cv2.drawContours(thresh, [contours[i]], 0, 255, -1)  # 进行像素填充处理
            kernel = np.ones((9, 9), np.uint8)  # 定义结构元素（kernal）的大小和形状，这里我们使用9*9的矩形结构元素
            dilated = cv2.dilate(thresh, kernel, iterations=2)

            # 提取图像骨架
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
            # if not end_points:  # 检查end_points是否为空
            #     continue  # 如果为空则跳过下面的循环

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
                    continue

            Max_suichang = max(List_suichang)
            # except Exception as e:
                # print('该图像处理失败')
                # continue
            real_suichang = Max_suichang * pixels_per_mm
            # results.append([filename, real_suichang, pixels_per_mm])
            # df = pd.DataFrame(results, columns=['文件名', '穗长', '转换比率'])
            # df.to_excel(output_file, index=False)

            # print(f"稻穗{filename}的穗长为：", real_suichang, 'mm', '转换比率:', {pixels_per_mm})
            # i = f"稻穗{filename}的穗长为：", real_suichang, 'mm', '转换比率:', {pixels_per_mm}
            # self.show_textedit_showdata.append(str(i))

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
                cv2.resize(skeleton, (240, 300))
                cv2.resize(img_uncut, (240, 300))
                result_img.save('./skeleton.jpg')  # 保存图片到指定路径
            pixmap_raw = QPixmap(raw_img)
            pixmap_res = QPixmap('./skeleton.jpg')
            pixmap_raw = pixmap_raw.scaled(self.show_label_window_raw_img.size(), Qt.KeepAspectRatio)
            pixmap_res = pixmap_res.scaled(self.show_label_window_panicle.size(), Qt.KeepAspectRatio)
            self.show_label_window_raw_img.setAlignment(Qt.AlignHCenter)  # 居中显示
            self.show_label_window_panicle.setAlignment(Qt.AlignHCenter)  # 居中显示
            self.show_label_window_raw_img.setPixmap(pixmap_raw)  # 显示原图
            self.show_label_window_panicle.setPixmap(pixmap_res)  # 显示结果图

            # 定义一个函数，根据序号生成两位字母

        def get_two_letters(j):
            # 如果序号小于26，直接返回一个大写字母
            if j < 26:
                return chr(ord('A') + j)
            # 否则，返回两个大写字母，第一个是商，第二个是余数
            else:
                return chr(ord('A') + j // 26 - 1) + chr(ord('A') + j % 26)

        results = []  # 存放数据的列表
        rectangles = []  # 此图片所有标注框的坐标信息
        lower_RGB, upper_RGB = (26, 109, 159), (134, 189, 214)  # 定义RGB阈值

        # # 进行距离矫正，找到合适的像素对真实世界转换比率
        # img_rgb = cv2.cvtColor(img_uncut, cv2.COLOR_BGR2RGB)  # 将图像转换成RGB颜色空间
        # mask_uncut = cv2.inRange(img_rgb, lower_red, upper_red)  # 根据颜色范围进行红色实心圆的掩码
        # kernal = np.ones((7, 7), np.uint8)  # 定义结构元素（kernal）的大小和形状，这里我们使用7x7的矩形结构元素
        # img_close = cv2.morphologyEx(mask_uncut, cv2.MORPH_CLOSE, kernal)  # 使用闭运算（close）来消除小物体或者去除物体中的小洞
        # contours, _ = cv2.findContours(img_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓

        image_name = os.path.splitext(os.path.basename(raw_img))[0]  # 去除后缀的图片名
        output_LW = self.detection_LW.main(cut_img)

        # 遍历output中的每个字典
        for item in output_LW:
            box = item['box']  # 获取'box'键对应的列表
            x1, y1, w, h = box  # 假设box列表的格式为[x1, y1, width, height]
            x2 = x1 + w  # 计算x2
            y2 = y1 + h  # 计算y2
            coordinate = (x1, y1, x2, y2)  # 组成坐标元组
            rectangles.append(coordinate)  # 将坐标添加到列表中

        # 指定目录，如果不存在就创建
        directory = os.path.join(sub_folder, image_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 遍历矩形列表，复制出子图，保存到文件夹中，按照图片名加上英文序列号命名
        for i, rectangle in enumerate(rectangles):
            padded_img = ImageOps.expand(img_cc.crop(rectangle), border=(15, 15, 15, 15), fill=1)  # 复制子图并添加 padding
            padded_img.save(os.path.join(directory, f"{image_name}{get_two_letters(i)}.jpg"))  # 保存子图到文件夹中，拼接子图的完整路径

        sub_save_folder = os.path.join(subfolder_process, image_name)
        if not os.path.exists(sub_save_folder):
            os.makedirs(sub_save_folder)
        List_chang = []
        List_kuan, List_kuanT5 = [], []
        List_ratio = []
        List_None = []
        sub_folder_path = os.path.join(sub_folder, image_name)

        for sub_filename in os.listdir(sub_folder_path):
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
        if len(List_chang) >= 7:
            A_chang = sum(List_chang) / len(List_chang)
            A_kuan = sum(List_kuan) / len(List_kuan)
            A_ratio = sum(List_ratio) / len(List_ratio)
            real_A_chang = A_chang * pixels_per_mm
            real_A_kuan = A_kuan * pixels_per_mm
            results.append([image_name, real_A_chang, real_A_kuan, A_ratio, pixels_per_mm])
            # print(f"文件{image_name}已经计算完成  粒长:{real_A_chang}  粒宽：{real_A_kuan}   长宽比:{A_ratio}   像素转换比率：{pixels_per_mm}")
            i_LW = [image_name, round(real_suichang, 1), round(real_A_chang, 1), round(real_A_kuan, 1), round(A_ratio, 1), round(pixels_per_mm, 2)]
            # i_LW = [image_name, self.real_suichang, real_A_chang, real_A_kuan, A_ratio, pixels_per_mm]
            row_count = self.show_textedit_showdata.rowCount()
            self.show_textedit_showdata.insertRow(row_count)
            print(i_LW)
            # i_LW_int = int(i_LW)
            # self.show_textedit_showdata.append(str(i_LW))
            for col, data in enumerate(i_LW):
                item = QtWidgets.QTableWidgetItem(str(data))
                self.show_textedit_showdata.setItem(row_count, col, item)
            # self.show_textedit_showdata.viewport().update()
        else:
            List_None.append(image_name)
        infer_img = cut_img.replace('cut', 'infer')
        pixmap_infer = QPixmap(infer_img)
        pixmap_infer = pixmap_infer.scaled(self.show_label_window_grain.size(), Qt.KeepAspectRatio)
        self.show_label_window_grain.setAlignment(Qt.AlignHCenter)  # 居中显示
        self.show_label_window_grain.setPixmap(pixmap_infer)  # 显示原图


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Form = QMainWindow()  # 使用 QMainWindow 替代 QWidget
    ui = Ui_Form()
    ui.setupUi(Form)

    # 获取屏幕的分辨率来计算窗口位置
    screen_resolution = app.desktop().screenGeometry()
    width, height = screen_resolution.width(), screen_resolution.height()

    # 计算窗口的 x 和 y 坐标使其居中
    x = (width - Form.width()) // 2
    y = (height - Form.height()) // 2

    Form.setGeometry(x, y, Form.width(), Form.height())  # 设置窗口位置
    Form.setWindowTitle("Extraction of rice panicle traits")  # 设置窗口标题
    Form.show()
    sys.exit(app.exec_())


