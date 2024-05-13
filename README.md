You can extract the traits of rice grains through this project.This project provides a software to extract the traits of rice grains. It also includes the scripts and neural networks required to use the software.

如何使用这个项目
首先你需要下载软件Panicle Analyzer，软件可以在这里获得：https://drive.google.com/file/d/1idG-fR5kl_vuKzSj5mTUOMDgD-m4gNRU/view?usp=drive_link
双击运行软件。软件有三个界面，分别是单张稻穗图像的性状提取界面、穗长批量提取界面和粒长粒宽批量提取界面。

单张图像的性状提取：
1.采集稻穗图像，使用图像采集设备采集稻穗图像。请注意，请使用黑色背景和红色标定物进行拍摄，样图如下![865 - 副本](https://github.com/SUNJHZAU/EOPT/assets/169641564/588c7461-d68c-4508-83e8-163a5044e7b9)
2.对稻穗图像进行裁剪并调整图像高宽比。我们提供了脚本帮助您完成此步骤：


你需要在github上下载yolov7项目之后，使用我们提供的权重对裁剪后的稻穗图像进行推理，以获得推理结果（.txt）。权重可以在这里获得：https://drive.google.com/file/d/1qeqP_ek3PXbiiShQ6i1vv-GoA5rAmcgO/view?usp=drive_link
