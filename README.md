
# Reducing-Colors-On-Image

"Reducing Colors On Image" I Used k-Mean Clustering Algorithm.so the some information about k-Mean Clustering Algorithm is this algorithm is an unsupervised learning algorithm that is used to solve the clustering problems in machine learning.

## what is k-Mean Clustering Algorithm?

K-Means Clustering is an Unsupervised Learning algorithm is providing the groups of the unlabeled dataset into different clusters and "K" defines the no. of pre-defined clusters and that they need to be created in the process. so, if K=2 is given so there will be two clusters and for K=3 is the given there will be three clusters

to perform k-Mean Clustering Algorithm steps are the following :-

Step-1: Select the number K to decide the number of clusters.

Step-2: Select random K points or centroids. (It can be other from the input dataset).

Step-3: Assign each data point to their closest centroid, which will form the predefined K clusters.

Step-4: Calculate the variance and place a new centroid of each cluster.

Step-5: Repeat the third steps, which means reassign each datapoint to the new closest centroid of each cluster.

Step-6: If any reassignment occurs, then go to step-4 else go to FINISH.

Step-7: The model is ready.


## Authors

- [@SiddharthNi](https://github.com/SiddharthNi)


## Installation

Install this my project by using jupyter notebook
when you installing jupyter notebook go on homepage of jupyter notebook on right side click on the "upload" and upload my"Reducing Colors On Image.ipynb"file. 


```bash
   install jupyter notebook
```
   matplotlib for ploting and showing image
```bash
   pip install matplotlib
```
   KMeans is a part of the Scikit-learn (sklearn) library it is in popular machine learning library in Python used for various tasks such as classification, regression, and clustering but mostly clustering.
```bash
   pip install scikit-learn
```

   "pip install scikit-image" is used  installing the scikit-image package and it is providing the functions and algorithms provided by scikit-image in  Python programming.
```bash
   pip install scikit-image
```
   
## Implementation

So the I make this project on jupyter notebook so  implementation steps are given below :-

step1:-import matplotlib.pyplot as plt this command is used for matploting and ".pyplot" for ploting in python
```bash
       import matplotlib.pyplot as plt
       %matplotlib inline

```

Step2:-"pip install scikit-image" is used installing the scikit-image package and it is providing the functions and algorithms provided by scikit-image in Python programming.
```bash
input:-pip install scikit-image
```
Step3:-in scikit image packaging we importing the read imaging and show image commands so this are given below
```bash
       from skimage.io import imread,imshow
       image = imread('siddharth.jpg')
       imshow(image)
```
    output:-so, this showing the image

Step4:-KMeans is a part of the Scikit-learn (sklearn) library it is in popular machine learning library in Python used for various tasks such as classification, regression, and clustering but mostly clustering.
```bash
from sklearn.cluster import KMeans
```
Step5:-its providing information about image shape.
```bash
image.shape
```
    output:-(1280, 1229, 3)

Step6:-we get reshape the image
```bash
X = image.reshape(-1,3)/255
```
Step7:-In KMeans we using clustering method.
```bash
kmeans = KMeans(n_clusters=3)
       X.shape
```
    output:-(1573120, 3)

Step8:-we using "Kmean.fit(x) for fiting of kn part in "X" so "X" is the image

```bash
       means.fit(X)
```
    output:-KMeans(n_clusters=3)

```bash
       kmeans.cluster_centers_
```
       
       output:-array([[0.87927523, 0.76902329, 0.68422498],
       [0.46021318, 0.18153355, 0.07161557],
       [0.50770244, 0.39749936, 0.41673808]])

```bash
        
       kmeans.labels_
```
       
       output:-array([0, 0, 0, ..., 1, 1, 1])
       
Step9:-kmeans providing the reshape of image and reducing colors on image
```bash
        new_X = kmeans.cluster_centers_[kmeans.labels_]
```
Step10:-providing reshape of image
```bash
       image = new_X.reshape(1280, 1229, 3)
```
Step11:-showing the reducing colors image
```bash
imshow(image)
```


    
## Screenshots
![IMG_20230422_212240](https://user-images.githubusercontent.com/116881073/233798903-d137b0b4-421c-4546-a6db-c9d86f563195.jpg)



## Lessons Learned

Image Preprocessing:- One of the first steps in the project would be to preprocess the image data to make it suitable for the clustering algorithm. This might include resizing the image, converting it to a suitable color space, and flattening the image data into a 2D array.

K-means Clustering:- The project would require knowledge of K-means clustering algorithm, how it works and how to implement it in Python. This includes setting the optimal value for K, initializing the centroids, and performing the clustering process.

Color Reduction:- The project would require reducing the number of colors in the image by mapping similar pixels to the same color cluster center. This could be done by replacing the pixel values with the centroid values of their respective clusters.

Performance Optimization:- The project might face some performance issues while clustering large images, so optimizing the algorithm's performance by using appropriate data structures and parallel processing techniques would be essential.

Some of the challenges one might face while building this project that are the following

Optimal Value of K:- Determining the optimal value of K for the clustering algorithm could be challenging. Several approaches, such as the elbow method or silhouette score, could be used to find the optimal value of K.

Overfitting:- Overfitting could be a concern, especially when the number of clusters is high. Using regularization techniques such as L2 normalization or early stopping could help mitigate overfitting



       



