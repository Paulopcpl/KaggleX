# **Possible Data Sources**

Below you can find some datasets that could be used to train models to perform *Automatic Licence Plates Recognition* (ALPR), *Super Resolution*, and *OCR*.

To train ALPR, the **RodoSol-ALPR Dataset** seems to be the best option (although we need to reach out for the authors for them to send us the dataset, it seems to be worth it).

The **Real Blur Dataset** seems to be a good option to train on Super Resolution. There are real and synthetic images in this dataset.

For text and numbers in images (aside from car plates) I found the higher number of datasets. The **SVHN (Street View House Numbers)** is the most famous dataset to work with house numbers. Another dataset that is interesting is the **MJSynth Dataset**, it has 9 million synthetic generated images with text. I indicated some other datasets that maybe you could find interesting, but these two seemed to be the two most promissing regarding numbers and texts in images (aside from car plates).

## **Automatic Licence Plates Recognition (ALPR) Datasets**
### [**Car License Plate Detection**](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)
This dataset contains 433 images with bounding box annotations of the car license plates within the image.

Annotations are provided in the PASCAL VOC format.

### [**UFPR-ALPR Dataset**](https://web.inf.ufpr.br/vri/databases/ufpr-alpr/)
This dataset, called UFPR-ALPR dataset, includes 4,500 fully annotated images (over 30,000 LP characters) from 150 vehicles in real-world scenarios where both the vehicle and the camera (inside another vehicle) are moving.

The images were acquired with three different cameras and are available in the Portable Network Graphics (PNG) format with a size of 1,920 × 1,080 pixels. The cameras used were: GoPro Hero4 Silver, Huawei P9 Lite, and iPhone 7 Plus.

\* *It is necessary to [ask for the dataset to the authors](https://web.inf.ufpr.br/vri/databases/ufpr-alpr/license-agreement/) through the use of academic credentials.*

### [**RodoSol-ALPR Dataset**](https://github.com/raysonlaroca/rodosol-alpr-dataset)
This dataset, called RodoSol-ALPR dataset, contains 20,000 images captured by static cameras located at pay tolls owned by the Rodovia do Sol (RodoSol) concessionaire, which operates 67.5 kilometers of a highway (ES-060) in the Brazilian state of Espírito Santo.

There are images of different types of vehicles (e.g., cars, motorcycles, buses and trucks), captured during the day and night, from distinct lanes, on clear and rainy days, and the distance from the vehicle to the camera varies slightly. All images have a resolution of 1,280 × 720 pixels.

An important feature of the proposed dataset is that it has images of two different LP layouts: Brazilian and Mercosur (to maintain consistency with previous works, we refer to “Brazilian” as the standard used in Brazil before the adoption of the Mercosur standard). All Brazilian LPs consist of three letters followed by four digits, while the initial pattern adopted in Brazil for Mercosur LPs consists of 3 letters, 1 digit, 1 letter and 2 digits, in that order. In both layouts, car LPs have the seven characters arranged in one row, whereas motorcycle LPs have three characters in one row and four characters in another. Even though these LP layouts are very similar in shape and size, there are considerable differences in their colors and also in the font of the characters.

\* *It is necessary to **ask for the dataset to the authors** (the instructions are on the link) through the use of academic credentials. But it seems to be worth it.* 

## **Super Resolution Dataset**

### [**Real Blur Dataset**](http://cg.postech.ac.kr/research/realblur/)
A large-scale dataset of real-world blurred images and ground truth sharp images for learning and benchmarking single image deblurring methods. To collect this dataset, the authors build an image acquisition system to simultaneously capture geometrically aligned pairs of blurred and sharp images, and develop a postprocessing method to produce high-quality ground truth images. The authors show that this dataset can significantly help to improve deblurring quality for real-world blurred images. 

## **Images with Text Data and Numbers**

### [**SVHN (Street View House Numbers)**](http://ufldl.stanford.edu/housenumbers/)
SVHN is a real-world image dataset for developing machine learning and object recognition algorithms with minimal requirement on data preprocessing and formatting. It can be seen as similar in flavor to MNIST (e.g., the images are of small cropped digits), but incorporates an order of magnitude more labeled data (over 600,000 digit images) and comes from a significantly harder, unsolved, real world problem (recognizing digits and numbers in natural scene images). SVHN is obtained from house numbers in Google Street View images. 

### [**COCO-Text: Dataset for Text Detection and Recognition**](https://vision.cornell.edu/se3/coco-text-2/)
The COCO-Text dataset contains 63,686 images with 145,859 cropped text instances. It is the first large-scale dataset for text in natural images and also the first dataset to annotate scene text with attributes such as legibility and type of text. However, no lexicon is associated with COCO-Text.

### [**The IIIT 5K-word dataset**](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset)
The IIIT 5K-word dataset is harvested from Google image search. Query words like billboards, signboard, house numbers, house name plates, movie posters were used to collect images. The dataset contains 5000 cropped word images from Scene Texts and born-digital images. The dataset is divided into train and test parts. This dataset can be used for large lexicon cropped word recognition. It is also provided a lexicon of more than 0.5 million dictionary words with this dataset.

### [**MJSynth Dataset**](https://www.robots.ox.ac.uk/~vgg/data/text/)
Provided by the University of Oxford, this word dataset has nearly 9 million synthetically generated images covering more than 90 thousand English language words.

### [**The Street View Text Dataset**](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)
The SVT dataset contains 350 images: 100 for training and 250 for testing. Some images are severely corrupted by noise, blur, and low resolution. Each image is associated with a 50 -word lexicon.

### [**MSRA Text Detection 500 Database (MSRA-TD500)**](http://www.iapr-tc11.org/mediawiki/index.php/MSRA_Text_Detection_500_Database_(MSRA-TD500))
An open-source database, the Text Detection dataset contains about 500 indoor and outdoor images of signboards, door plates, caution plates, and more.