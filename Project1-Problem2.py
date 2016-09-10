import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import Image


#Load Image
image = cv2.imread('C:\\Users\\Archith\\Desktop\\CVIP\\Project1\\butterfly.jpg')
plt.imshow(image)
plt.show()

#Convert Image into Grayscale and display
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_image.jpg',gray_image)
plt.imshow(gray_image, cmap = cm.Greys_r)
plt.show()


#Get Image Parameters
size = gray_image.size
height  = gray_image.shape[0]
width = gray_image.shape[1]
image2 = np.zeros((height,width))


#Initialize Data Structures needed for the Computations
values = np.arange(0,256)
Count = [0] * 256
Hc = [0] * 256
H = [0] * 256
Hnew = [0] *256
Gray_Levels = []
New_Gray_Levels = []

# Image Histogram
for i in range(height): 
    for j in range (width): 
      value = image[i,j,2]
      Gray_Levels.append(value)
      H[value] = H[value] +1
           

plt.figure('Image Histogram')
plt.xlabel('Pixels')
plt.ylabel('Pixels Count')
plt.title('Grayscale Image Histogram')
plt.plot(values,H)
plt.show()


# Cumulative Histogram
for x in range(height):
   for y in range(width):
        p = image[x,y,2]             
        Hc[0] =H[0]
        New_Gray_Levels.append(Hc[p])       
        Hc[p] = Hc[p-1] + H[p]
       
#Applying the Transformation Function      
for x in range(len(Hc)):
    Tp = round ( (255*Hc[p]) / (height*width) )
    New_Gray_Levels.append(Tp)     
Final = np.asarray(New_Gray_Levels)
Final = Final.astype(int)


plt.figure('Cumulative Histogram')
plt.xlabel('Pixels')
plt.ylabel('Pixels Count')
plt.title('Cumulative Histogram')
plt.plot(values,Hc)
plt.show()

#temporary counter used to replace current pixel positon with the new pixel from New_Gray_Levels array and appending to the new image
count = 0

for a in range(height):
   for b in range(width):       
        image2[a,b] = Final[count]        
        count = count+1
        
#for i in range(height): 
#    for j in range (width): 
#      value1 = image2[i,j,2]
#     
#      Hnew[value1] = Hnew[value1] +1
#           
#
#plt.figure('Image Histogram')
#plt.xlabel('Pixels')
#plt.ylabel('Pixels Count')
#plt.title(' New Grayscale Image Histogram')
#plt.plot(values,H)
#plt.show()
        
      




cv2.imshow('Equalized Image',image2)   
cv2.imwrite('out.png', image2)     


vis = np.concatenate((gray_image, image2), axis=1)
cv2.imwrite('out1.png', vis)  
      



















