import cv2
import numpy as np
import math

#Code to Read image
image = cv2.imread('C:\Users\Archith\Desktop\CVIP\lena_gray.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow("full image", image)
size  = image.size

# Declaring Filters Gx and Gy
Gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
Gy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

#Declaring Filters for ID 
Gx1 = np.array([[1],[2],[1]])
Gx2 =np.array([-1,0,1])
Gy1 = np.array([[-1],[0],[1]])
Gy2 = np.array([1,2,1])

# Computing the Magnitude Filter
Magnitude = np.sqrt(Gy**2 + Gx**2)

#Code for Image padding 
Padding = np.zeros((514,514),dtype='uint8')
Padding[1:513, 1:513] = image
height  =  Padding.shape[0] -1
width = Padding.shape[1] -1

# Initializing two images with zeros for 2D
image2 = np.zeros((512,512))
image3 = np.zeros((512,512))

#Initializing two images with zeros for ID
image4 = np.zeros((512,512)) 
image5 = np.zeros((512,512)) 

#Initializing an Image with zeros for Gradient Magnitude Image
[height1, width1] = image3.shape
Magimage = np.zeros((height1,width1))

#---------------------------END INITIALIZATION-----------------------------------------

# Code to get Gradient Image Gx
for a in range(1,height):
    for b in range(1,width):
            sum1  = sum(Padding[a-1:a+2,b-1:b+2]*Gx)
            Finalsum = sum(sum1)
            image2[a-1,b-1] = Finalsum             
image2/= image2.max()
cv2.imshow('Gradient Image Gx',image2)
cv2.imwrite("Gx.jpg",image2)

#Performing 1D Convolution to get the Gradient Image Gx
for a in range(1,height):
    for b in range(1,width):  
        p  = ((Padding[a-1:a+2,b-1:b+2]*Gx1).sum(axis = 0))
        p1 = p*Gx2
        Finalsum = sum(p1)
        image4[a-1,b-1] = Finalsum   
image4 /=image4.max()
cv2.imshow('Gx after 1D Convolution',image4)
cv2.imwrite('1DGx.jpg',image4)
#------------------------------------END------------------------------------------------


#Code to get Gradient Image Gy
for i in range(1,height):
    for j in range(1,width):
            sum1  = sum(Padding[i-1:i+2,j-1:j+2]*Gy)
            Finalsum = sum(sum1)
            image3[i-1,j-1] = Finalsum
             
image3/= image3.max()
cv2.imshow('Gradient Image Gy',image3)
cv2.imwrite('Gy.jpg',image3)

#Performing 1D Convolution to get the Gradient Image Gy
for i in range(1,height):
    for j in range(1,width):     
        p  = ((Padding[i-1:i+2,j-1:j+2]*Gy1).sum(axis = 0))
        p1 = p*Gy2
        Finalsum = sum(p1)
        image5[i-1,j-1] = Finalsum   
image5 /=image5.max()
cv2.imshow('Gy after 1D Convolution',image5)
cv2.imwrite('1DGy.jpg',image5)
#------------------------------------------------END-------------------------------------  

                                                                              
#Code to Compute the Gradient Magnitude Image
for x in range(height1):
    for y in range(width1):
        Gxsquare = math.pow(abs(image2[x,y]), 2)
        Gysquare = math.pow(abs(image3[x,y]), 2)
        Magimage[x,y] = math.sqrt(Gysquare+Gxsquare);
        Magimage[x,y]=Magimage[x,y]/255;
Magimage /= Magimage.max()
cv2.imshow('Gradient Magnitude Image', Magimage)
cv2.imwrite('Gmag.jpg',Magimage)
#----------------------------------------------------END---------------------------------     