# **Vehicle detection**

This is the fifth project I am doing as part of Udacity's Self-Driving-Car Nanodegree.

**The goals/steps of this project are the following:**
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.



[//]: # (Image References)

[image1]: ./output_images/hog_car.png "HOG Feature example"
[image2]: ./output_images/hog_car_Cb.png "HOG Feature example"
[image3]: ./output_images/hog_car_Cr.png "HOG Feature example"
[image4]: ./output_images/hog_non_car.png "HOG Feature example"
[image5]: ./output_images/hot1.png "Heatmap example"
[image6]: ./output_images/hot4.png "Heatmap example"
[image7]: ./output_images/hot5.png "Heatmap example"
[image8]: ./output_images/hot6.png "Heatmap example"
[image9]: ./output_images/test1.png "Test image 1"
[image10]: ./output_images/test2.png "Test image 2"
[image11]: ./output_images/test3.png "Test image 3"
[image12]: ./output_images/test4.png "Test image 4"
[image13]: ./output_images/test5.png "Test image 5"
[image14]: ./output_images/test6.png "Test image 6"
[image15]: ./output_images/search_grid.png "Sliding window grid"

# Report
### Writeup & Project Files

#### 1. Writeup
You're reading it! I will examine below the [rubric points](https://review.udacity.com/#!/rubrics/513/view) individually and describe how I addressed each point in my implementation.


#### 2. Project Files

You can find all project files in this [Github Repository](https://github.com/thoomi/vehicle-detection).

My project includes the following files and folders:
* [src/](https://github.com/thoomi/vehicle-detection/tree/master/src) containing all coded script files including the whole processing pipeline
* [output_images/](https://github.com/thoomi/vehicle-detection/tree/master/output_images) containing pipeline images of each individual step
* [trained_classifier.p](https://github.com/thoomi/vehicle-detection/blob/master/trained_classifier.p) containing the trained SVM model and the trained data scaler
* [parameter_experiments.ipynb](https://github.com/thoomi/vehicle-detection/blob/master/parameter_experiments.ipynb) a IPython notebook to evaluate and test various aspects of the pipeline easier and to generate example images
* [writeup_report.md](https://github.com/thoomi/vehicle-detection/blob/master/writeup_report.md)

---

### Histogram of Oriented Gradients (HOG)

#### 1. Feature extraction

The code for this step is located in the file `FeatureExtractor.py`. Within the lines 21-25 i converted the image color space to YCrCb which is found by try-and-error to work better for this kind of classification.

I found the following HOG extractor parameters to work best for me:

* Number of Orientations: **9**
* Pixel per cell: **16**
* Cells per block: **2**

There are example feature visualizations below for each color channel and an additional visualization of a non-car feature combination.

**Car-Features Y-Channel**
![Example of hog features][image1]

**Car-Features Cr-Channel**
![Example of hog features][image2]

**Car-Features Cb-Channel**
![Example of hog features][image3]

**Non-Car-Features Y-Channel**
![Example of hog features][image4]

#### 2. Train classifier

The training steps of the SVM are located within the file `trained_classifier.py`. The lines 23-32 contain code for loading the training images from the various provided datasets. All of them are combined and devided into the classes **car** and **not-car**. After that, the features for all of the provided images are extracted within the lines 61-62. This features are normalized by a fitted StandardScaler in the line 67. For the actual training process the data is splitted 80/20 into a training and test set. Unfortunately the evaluated accuracy value doesn't provide much insight about the classifiers performance on real world images.

---

### Sliding Window Search

#### 1. Overlapping sliding windows

The sliding window process is devided into the three window sizes 48, 64 and 128. Each of them are restricted to a specific image area where cars of that size should appear. I chose a sliding overlap of 70% in order to gain high confidence detections if there are a lot of bounding boxes within an area.

![Sliding window grid][image15]

#### 2. Example and Performance

---


### Result video

#### 1. Project Video Result

#### 2. False Positive rejection

---


### Discussion & Reflection


---


### Appendix


#### Blogs & Tutorials


#### Tools:
[tool01]: https://www.python.org/
[tool02]: http://opencv.org/

 - [Python][tool01]
 - [OpenCV][tool02]
