# real-time-ai
Realtime Artificial Intelligence (CS494 @UTK)

This repository outlines my journey experimenting with real time artificial intelligence. If you look through the presentations, you can see that I begin my experimentation with keras, implementing covolutional neural networks and performing image degradation (gaussian blur) in an attempt to improve model accuracy. Furthermore, I tested performance between multi-class classification models vs binary-class classification models to examine generalization within models. I have a few images and some code within the cifar folder. 

As the weeks progressed, I moved towards real time object detection with the aid of [YOLO](https://pjreddie.com/darknet/yolo/) to test object detection on a variety of objects such as people, bicycles, cars, bottles, and birds as outlined in the week 7-8 presentation. I learned about various modes of training i.e. on a cluster of computers, GPU, and discussed relative performance on an FPGA. I have added some demos that show the results of the object detection model that I trained to the demos directory.

For the most part, I utilized the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [STL-10](https://cs.stanford.edu/~acoates/stl10/) datasets for image classification. Then, I used the [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset for the YOLO object-detection models.
