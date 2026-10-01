import cv2
import numpy as np
import matplotlib.pyplot as plt

a = np.arange(10) #naredi tabelo 10 elementov do 9
[item**2 for item in a] #naredi kvadrate od a

img = cv2.imread('./assignment1/images/bird.jpg')
img.shape
plt.imshow(img)
plt.show() #prikaze sliko