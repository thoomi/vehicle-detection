# **Vehicle detection**

This is the fifth project I am doing as part of Udacity's Self-Driving-Car Nanodegree.

**The goals/steps of this project are the following:**
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

#### Preview:

[![Project video thumbnail][image16]](./result_project_video.mp4?raw=true)


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
[image16]: ./output_images/final_output_sample.gif "Final output"

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


Please not that all following file references refer to files within the folder `src/VehicleDetector` as this is the primary content for this project.

---

### Histogram of Oriented Gradients (HOG)

#### 1. Feature extraction

The code for this step is located in the file `FeatureExtractor.py`. Within the lines 21-25, I converted the image color space to YCrCb which is found by try-and-error to work better for this kind of classification.

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

The training steps of the SVM are located within the file `trained_classifier.py`. The lines 23-32 contain code for loading the training images from the various provided datasets. All of them are combined and divided into the classes **car** and **not-car**. After that, the features for all of the provided images are extracted within the lines 61-62. These features are normalized by a fitted StandardScaler in line 67. For the actual training process, the data is split 95/5 into a training and test set. Unfortunately, the evaluated accuracy value doesn't provide much insight into the classifiers performance on real world images.

---

### Sliding Window Search

#### 1. Overlapping sliding windows

The sliding window process is divided into the three window sizes 48, 64 and 96. Each of them is restricted to a specific image area where cars of that size should appear. I chose a sliding overlap of 80% in order to gain high confidence detections if there are a lot of bounding boxes within an area.

![Sliding window grid][image15]

#### 2. Example and Performance

In order to optimize the performance of the sliding window approach, I restricted the search area to be between 400 to 656 px in y-direction. Additionally, instead of the sklearn hog function, I used the OpenCV HOGDescriptor which utilizes the GPU to speed up the feature extraction process. See `FeatureExtractor.py` lines 62-84 for the implementation.

The first image below shows an example of all the detected windows which are combined into one at the end. The second image doesn't have a car in it, thus there is no bounding box.
![Test image][image9]
![Test image][image10]

---


### Result video
[videothumb1]: ./output_images/final_output.png "Final example 1"

#### 1. Project Video Result

[![Project video thumbnail][videothumb1]](./result_project_video.mp4?raw=true)


#### 2. False Positive rejection

Because the classifier is not perfect and might fire a detection even if there is no car in that area, we need a way to reject those false positives. In order to do that, I created a buffered heatmap which can be found in the file `BufferedHeatmap.py`. It saves heatmaps created in each frame over a period of **12** frames and applies a threshold on the combined version. That thresholded heatmap is fed into the ndimage `label` function which then returns high confidence detection areas.

See the image below for an example:

![Hot image][image7]

---


### Discussion & Reflection

This has been one of the hardest projects I did during the first term. Selecting features and training the linear classifier seems like an easy task, but it has been quite hard to find a good combination to minimize false positives. The algorithm is very likely to fail on roads which are not even because the search window is too restricted. And it still is to slow to be used in a real world detection scenario.

I am unfortunately not completely satisfied with the current implementation and this needs further investigation from my side. Besides, I had a lot of fun discovering and implementing those standard computer vision techniques. I feel like I have learned various approaches, I am confident to apply them and to be able and ready to advance on this path.

---


### Appendix


#### Tools:
[tool01]: https://www.python.org/
[tool02]: http://opencv.org/

 - [Python][tool01]
 - [OpenCV][tool02]
