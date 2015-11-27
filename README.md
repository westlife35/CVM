>Those are the implementations of algorithms from the book [Computer vision:models, learning and inference](http://computervisionmodels.com/) 

> * platform:win7+VS2012
> * the result of every chapter will show in the floder "Result"

>Content

> * Chapter 4 (Probability models)
    * __curve fitting of normal distribution and category distribution__
    * void MleNorm();
    * void MapNorm();
    * void ByNorm();
    * void MleCat();
    * void MapCat();
    * void ByCat();
    
    
> * Chapter 6 (Probability models)
    * __character recognition__ 
    * train image : 8422 images
    * number of pattern : 10(0~9)
    * feature : resize and equalize the image and use the pixels` number vecter as the feature directly
    * pre-handle: resize and histogram equalization
    * train accuracy: 94.775%(resize:15*9),93.7182%(3*6,faster)
    
    
> * Chapter 7 (Expectation Maximization)
    * __fitting of normal distribution with latent variance__ 
    * the sample use the independent normal distribution with dimension two and the total number of center is two which is easy to show in a coordinate.
    * there is also a comparision between the train data generated from the code matlab and the code cpp, and obviously the data from cpp code is pseudorandom, but it will not influence the result of the algorithm


> * Chapter 8 (Regresion Model)
    * accomplish the function of matrix computation by myself but it works not that efficiently
    * do not find too much highlight of the algorithm and my teacher tells us it is not that widely used like SVM
    

###Some Info

>Author: Ce Qi

>University: BUPT

>Function：Solution to the CVM

>Date: 2015-10-13

>Email:qi_ce_0@163.com

>Contact me if you have any questions