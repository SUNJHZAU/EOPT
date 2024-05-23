You can extract the traits of rice grains through this project.This project provides a software to extract the traits of rice grains. It also includes the scripts and neural networks required to use the software.

How to use this project

First you need to download the software Panicle Analyzer, the link to the software is here: [[https://drive.google.com/file/d/1idG-fR5kl_vuKzSj5mTUOMDgD-m4gNRU/view?usp=drive_link](https://drive.google.com/file/d/1YO7lHqfQT0VRAmMYbB7tP16cDR38O6wu/view?usp=drive_link)](https://drive.google.com/file/d/1YO7lHqfQT0VRAmMYbB7tP16cDR38O6wu/view?usp=drive_link)
Double click to run the software. The software has three interfaces, namely, the trait extraction interface for single panicle images, the batch extraction interface for panicle length and the batch extraction interface for grain length and width.

Trait extraction from a single image:

	1.Capture images of rice panicles. Capture images of the rice panicles using an image capture device. Please note that the image must be taken with a black background and a red calibrator, as shown in the sample image below![865 - 副本](https://github.com/SUNJHZAU/EOPT/assets/169641564/588c7461-d68c-4508-83e8-163a5044e7b9)
 
	2.Crop the panicle image and adjust the image aspect ratio. Just crop the image of the panicle until only the body of the panicle remains.
	
	3.After the operator selects the files required for extracting the traits in turn, click 'Start analysis' to display the processing results and images. The specific operation steps are as follows:
	
		Select Cropped Image: Click the corresponding button to select the cropped image of the panicle.
	 
		Select Original Image: Click the corresponding button to select the original panicle image.
	 
		Selection of Subgraph Saving Path: Click on the corresponding button to select the path where the grain image is to be saved.Do not click on the button to select the txt file.
	 
		Select processing sub-image save path: click the corresponding button to select the path to save the processed grain image.
	 
		Extracting traits and displaying results: click on the corresponding button to start extracting traits and displaying the results on the right hand side of the screen.


Batch extraction of panicle length:

	Click the 'Panicle length' button to switch to the appropriate screen. By this point, we have defaulted that you have captured and cropped the panicle image.
	
	After the operator selects the file directories required for extracting panicles in turn, click 'Start analysis' to display the processing results. The specific operation steps are as follows:
 
	Select Cropped Image: Click the corresponding button to select the directory where the cropped image of the panicle is located.
 
	Selection of original panicle image: click the corresponding button to select the directory where the original panicle image is located.
 
	Select the path to save the image of the main path of the panicle skeleton: click on the corresponding button to select the folder where the main path of the panicle is saved.
 
	Select data result file: Click the corresponding button to select the table file where the panicle data is saved.
 
	Start extraction of panicles: Click on the 'Start analysis' button to start extraction of panicles.
 
	Stop processing: Click on the 'stop' button to stop processing the image of the panicle.
 	


Batch extraction of grain length and width:

	Click on the 'Grain length, Grain width' button to switch to the corresponding interface.
	
	After the operator selects the file directories required for extracting panicles in turn, click 'Start analysis' to display the processing results. The specific operation steps are as follows:
	
	Select Cropped Image: Click the corresponding button to select the directory where the cropped image of the panicle is located.
	
	Selection of original panicle image: click the corresponding button to select the directory where the original panicle image is located.
	
	Selection of Subgraph Saving Path: Click on the corresponding button to select the path where the grain image is to be saved.Do not click on the button to select the txt file.
	
	Select processing sub-image save path: click the corresponding button to select the path to save the processed grain image.
	
	Selection of data result file: Click the corresponding button to select the table file in which the grain length and width data will be saved.
	
	Start extraction of grain length and width: Click on the 'Start analysis' button to start extraction of grain length and width.
	
	Stop processing: Click on the 'stop' button to stop processing the image of the panicle.



If you do not want to capture new panicle images and crop images. We have provided some original and cropped images of the images for you to useDemos can be found here:https://drive.google.com/drive/folders/1S7BHcjutJ-wtHdn-jR92YGB4lBSxWb-s?usp=drive_link。


In addition, we have made all of the code used in this project publicly available for subsequent work. The code about the software Panicle Analyzer can be found in panicle_length.py. The core code about panicle length extraction can be found in panicle_length.py. Code for grain length and width extraction can be found in LWtrait.py.
