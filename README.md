You can extract the traits of rice grains through this project.This project provides a software to extract the traits of rice grains. It also includes the scripts and neural networks required to use the software.

如何使用这个项目

首先你需要下载软件Panicle Analyzer，软件可以在这里获得：https://drive.google.com/file/d/1idG-fR5kl_vuKzSj5mTUOMDgD-m4gNRU/view?usp=drive_link
双击运行软件。软件有三个界面，分别是单张稻穗图像的性状提取界面、穗长批量提取界面和粒长粒宽批量提取界面。

单张图像的性状提取：

	1.采集稻穗图像，使用图像采集设备采集稻穗图像。请注意，请使用黑色背景和红色标定物进行拍摄，样图如下![865 - 副本](https://github.com/SUNJHZAU/EOPT/assets/169641564/588c7461-d68c-4508-83e8-163a5044e7b9)
 
	2.对稻穗图像进行裁剪并调整图像高宽比。我们提供了脚本帮助您完成此步骤：crop_img.py。
	
	3.你需要在github上下载yolov7项目之后，使用我们提供的权重对裁剪后的稻穗图像进行推理，以获得推理结果（.txt）。权重可以在这里获得：https://drive.google.com/file/d/1qeqP_ek3PXbiiShQ6i1vv-GoA5rAmcgO/view?usp=drive_link 。关于如何使用yolov7进行推理，请详细阅读yolov7的官方文档。
	
	4.操作人员依次选择提取性状所需的文件后，点击‘Start analysis’即可展示处理结果和图像。具体操作步骤如下：
	
		选择裁剪图像：点击相应按钮选择裁剪后的稻穗图像
	 
		选择原始图像：点击相应按钮选择原始稻穗图像
	 
		选择推理结果文件：点击相应按钮选择包含深度学习预测结果的文件 （.txt）
	 
		选择子图保存路径：点击相应按钮选择籽粒图像保存的路径
	 
		选择处理子图保存路径：点击相应按钮选择处理后的籽粒图像保存的路径
	 
		提取性状并展示结果：点击相应按钮开始提取性状并且将结果展示右侧界面中


穗长的批量提取：

	点击‘Panicle length’按钮转换到相应界面。到这里，我们默认你已经采集了稻穗图像并进行了裁剪。
	
	操作人员依次选择提取穗长所需的文件目录后，点击‘Start analysis’即可展示处理结果。具体操作步骤如下：
 
	选择裁剪图像：点击相应按钮选择裁剪后的稻穗图像所在目录
 
	选择原始稻穗图像：点击相应按钮选择原始稻穗图像所在目录
 
	选择稻穗骨架主路径图像的保存路径：点击相应按钮选择保存稻穗骨架主路径的文件夹
 
	选择数据结果文件：点击相应按钮选择穗长数据保存的表格文件
 
	开始提取穗长：点击‘Start analysis’按钮开始提取穗长
 
	停止处理：点击‘stop’按钮停止处理稻穗图像
 	


粒长粒宽的批量提取：

	点击‘Grain length、Grain width’按钮转换到相应界面。
	
	操作人员依次选择提取穗长所需的文件目录后，点击‘Start analysis’即可展示处理结果。具体操作步骤如下：
	
	选择裁剪图像：点击相应按钮选择裁剪后的稻穗图像所在目录
	
	选择原始稻穗图像：点击相应按钮选择原始稻穗图像所在目录
	
	选择预测结果：点击相应按钮选择保存深度学习预测结果的文件夹
	
	选择子图保存路径：点击相应按钮选择籽粒图像保存的路径
	
	选择处理子图保存路径：点击相应按钮选择处理后的籽粒图像保存的路径
	
	选择数据结果文件：点击相应按钮选择粒长、粒宽数据保存的表格文件
	
	开始提取粒长粒宽：点击‘Start analysis’按钮开始提取粒长和粒宽
	
	停止处理：点击‘stop’按钮停止处理稻穗图像


如果你没有稻穗图像，也不想下载yolov7，我们提供了一个demo文件夹，其中包含了稻穗图像和yolov7的推理结果，以帮助你快速学习使用软件。demo在这里找到：https://drive.google.com/drive/folders/1S7BHcjutJ-wtHdn-jR92YGB4lBSxWb-s?usp=drive_link。


