#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import scipy
from scipy import ndimage, signal

import cv2                  
from skimage import io, color, filters, exposure, util, morphology, restoration
from skimage.util import random_noise
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.io import loadmat

import matplotlib.pyplot as plt
import matplotlib.image as matimg


# In[21]:


#1
def calculate(original, processed):
    psnrCal = psnr(original, processed, data_range=255)
    ssimCal = ssim(original, processed, data_range=255, win_size=3)
    print(f"PSNR: {psnrCal} dB")
    print(f"SSIM: {ssimCal} dB")
    return psnrCal, ssimCal
#Υπολογισα το psnr με ευρος 255
#Υπολογισα το ssim με 'παραθυρο' 3Χ3




img = cv2.imread('pepper.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Φόρτωσα την εικονα και μετέτρεψα τον χρωματικο χώρο απο BGR σε RGB.

img = img.astype(np.float32) / 255.0   
# Κανονικοποιηση για αποφυγη υπαρχειλισης στις πραξεις που θα χαλασει την εικονα με ευρος απο 0.0 εως 1.0.


P_signal =np.mean(img**2)

#Υπολογιζω την ισχυ του σηματος οπου ειναι ο μεσος ορος των πιξελ στο τετραγωνο


SNRdB = 15
P_noise = P_signal / (10**(SNRdB / 10))
# Ορισα το SNR και υπολογισα τη τυπικη αποκλιση του θορύβου
sigma = np.sqrt(P_noise)


G_noise = np.random.normal(0, sigma, img.shape)

noisy_G = img + G_noise


noisy_G = np.clip(noisy_G, 0, 1)
#Φροντιζω να μεινει στο οριο [0,1] γιατι χωρις αυτο μου κατεστρεφε την εικονα


mean_G = cv2.blur((noisy_G*255).astype(np.uint8), (3,3))

median_G = cv2.medianBlur((noisy_G*255).astype(np.uint8), 3)
#Επαναφερα την εικονα σε μορφη unit8
#Εφάρμοσα φιλτρο κινούμενου μεσου με παράθυρο 3 επι 3.
#Εφάρμοσα φίλτρο διαμέσου με μέγεθος 3.

plt.figure(figsize=(8,8))
    
plt.subplot(2,2,1)
plt.imshow(img)
plt.title("Original")
plt.axis('off')


plt.subplot(2,2,2)
plt.imshow(noisy_G)
plt.title("Gaussian")
plt.axis('off')


plt.subplot(2,2,3)
plt.imshow(mean_G)
plt.title("Mean Filter")
plt.axis('off')


plt.subplot(2,2,4)
plt.imshow(median_G)
plt.title("Median Filter")
plt.axis('off')


plt.show()

print('After mean filter calculations')
calculate((img* 255).astype('uint8'),mean_G);
print('After median filter calculations')
calculate((img* 255).astype('uint8'),median_G);
#Μετετρεψα αυτα που δεν ειναι unit8


# In[23]:


#1.2       
noisy_SP = random_noise(img, mode='s&p', amount=0.2)
# Πρόσθεσα θόρυβο Salt & Pepper στην αρχική εικόνα με 20% 
mean_SP = cv2.blur((noisy_SP*255).astype(np.uint8), (3,3))

median_SP = cv2.medianBlur((noisy_SP*255).astype(np.uint8), 3) 


# Εμφάνιση
plt.figure(figsize=(8,8))
    
plt.subplot(2,2,1)
plt.imshow(img)
plt.title("Original")
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(noisy_SP)
plt.title("Salt & Pepper")
plt.axis('off')


plt.subplot(2,2,3)
plt.imshow(mean_SP)
plt.title("Mean Filter")
plt.axis('off')


plt.subplot(2,2,4)
plt.imshow(median_SP)
plt.title("Median Filter")
plt.axis('off')
plt.show()

print('After mean filter :')
calculate((img* 255).astype('uint8'),mean_SP);
print('After median filter :')
calculate((img* 255).astype('uint8'),median_SP);


# In[24]:


#1.3

noisy_GSP = random_noise(noisy_G, mode='s&p', amount=0.2)
# Δημιουργησα τον συνδυασμένο θορυβο κανοντας Salt & Pepper πάνω στην ηδη υπάρχουσα εικόνα με Gaussian θορυβο (noisy_G).
mean_GSP = cv2.blur((noisy_GSP*255).astype(np.uint8), (3,3))

median_GSP = cv2.medianBlur((noisy_GSP*255).astype(np.uint8), 3) 

medianFirst = cv2.blur(median_GSP, (3,3))
#Εκανα τη διαδοχική εφαρμογή φίλτρων με τη σειρά: Median -> Mean.

meanFirst = cv2.medianBlur(mean_GSP, 3)
#Εκανα τη διαδοχική εφαρμογή φίλτρων με τη σειρά: Mean -> Median.




plt.figure(figsize=(8,12))
    
plt.subplot(3, 2, 1)
plt.imshow(img)
plt.title("Original")
plt.axis('off')

plt.subplot(3, 2, 2)
plt.imshow(noisy_GSP)
plt.title("Gaussian και Salt & Pepper") 
plt.axis('off')

# Σειρά 2: Μεμονωμένα Φιλτρα
plt.subplot(3, 2, 3)
plt.imshow(mean_GSP)
plt.title("Mean")
plt.axis('off')

plt.subplot(3, 2, 4)
plt.imshow(median_GSP)
plt.title("Median")
plt.axis('off')

# Σειρά 3: Πρωτα Mean -> Median μετα Median->Mean
plt.subplot(3, 2, 5)
plt.imshow(meanFirst)
plt.title("Mean -> Median ")
plt.axis('off')

plt.subplot(3, 2, 6)
plt.imshow(medianFirst)
plt.title("Median -> Mean") 
plt.axis('off')

plt.show()


print('After putting mean filter first :')
calculate((img* 255).astype('uint8'),meanFirst);
print('After putting median filter first :')
calculate((img* 255).astype('uint8'),medianFirst);


# In[25]:


#2

#Εφτιαξα συναρτηση εμφανισης plot για σωσιμο χρονου
def showImgHist(img):
    plt.figure(figsize=(10,2))
    
    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("original")
    plt.axis('off')
    
    plt.subplot(1,2,2)
    
    hist = cv2.calcHist([img], [0], None, [256], [0,256])
    # Υπολογισα το ιστογράμμα για 0-255 τιμες φωτεινοτητας (εφοσον ειναι ασπρομαυρη)
    plt.plot(hist)
    plt.title("Histogram")
    plt.xlabel("Pixel Values")
    plt.ylabel("Frequency")
    #
    
    plt.show()


# In[26]:


# Φορτωσα τις εικονες σε grayscale
road1 = cv2.imread('dark_road_1.jpg', cv2.IMREAD_GRAYSCALE)
road2 = cv2.imread('dark_road_2.jpg', cv2.IMREAD_GRAYSCALE)
road3 = cv2.imread('dark_road_3.jpg', cv2.IMREAD_GRAYSCALE)

showImgHist(road1)
showImgHist(road2)
showImgHist(road3)


# In[28]:


road1Eq = cv2.equalizeHist(road1)
road2Eq = cv2.equalizeHist(road2)
road3Eq = cv2.equalizeHist(road3)
# Εφάρμοσα την τεχνική της ολικης εξισωσης ιστογράμματος

def showEqHist(original, equalized):
    plt.figure(figsize=(10,7))
    
    plt.subplot(2,2,1)
    plt.imshow(original, cmap='gray')
    plt.title("Αρχικη")
    plt.axis('off')
    
    plt.subplot(2,2,2)
    histOr = cv2.calcHist([original], [0], None, [256], [0,256])
    plt.plot(histOr)
    plt.title("Original's Histogram")
    
    plt.subplot(2,2,3)
    plt.imshow(equalized, cmap='gray')
    plt.title("Global Eq")
    plt.axis('off')
    
    plt.subplot(2,2,4)
    histEq = cv2.calcHist([equalized], [0], None, [256], [0,256])
    plt.plot(histEq)
    plt.title("Histogram Of Globa Eq")
    
    plt.tight_layout()
    plt.show()

    
    

showEqHist(road1, road1Eq)
showEqHist(road2, road2Eq)
showEqHist(road3, road3Eq)


# In[29]:


cla8  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cla16 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
#Εφαρμοσα CLAHE με 2 παραμετρους

road1_CL8  = cla8.apply(road1)
road1_CL16 = cla16.apply(road1)

road2_CL8  = cla8.apply(road2)
road2_CL16 = cla16.apply(road2)

road3_CL8  = cla8.apply(road3)
road3_CL16 = cla16.apply(road3)

# Εφτιαξα συναρτηση εμφάνισης: αρχικη , ολική , clahe
def showAllEq(original, oliki_eq , clahe_eq,name):
    plt.figure(figsize=(16,4))
    
    plt.subplot(1,4,1)
    plt.imshow(original, cmap='gray')
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(1,4,2)
    plt.imshow(oliki_eq, cmap='gray')
    plt.title("Global Eq")
    plt.axis('off')
    
    plt.subplot(1,4,3)
    plt.imshow(clahe_eq, cmap='gray')
    plt.title(name)
    plt.axis('off')

    plt.subplot(1,4,4)
    plt.hist(clahe_eq.ravel(), 256, [0, 256], color='black')
    plt.title("Histogram")
    plt.xlabel('Brightness')
    plt.ylabel('Pixels')
    
    
    
    plt.tight_layout()
    plt.show()


#Εφτιαξα συναρτηση συγκρισης των ιστογραμματων μόνο
def compareHist (original, ImgEq , CL8  , CL16 ):

    plt.figure(figsize=(20,4))
    
    plt.subplot(1,4,1)
    plt.imshow(original, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1,4,2)
    plt.hist(ImgEq.ravel(), 256, [0, 256], color='black')
    plt.title("Histogram Of Global Eq")
    plt.xlabel('Brightness')
    plt.ylabel('Pixels')
    

    plt.subplot(1,4,3)
    plt.hist(CL8.ravel(), 256, [0, 256], color='black')
    plt.title("Histogram Of Clahe 8")
    plt.xlabel('Brightness')
    plt.ylabel('Pixels')
    

    plt.subplot(1,4,4)
    plt.hist(CL16.ravel(), 256, [0, 256], color='black')
    plt.title("Histogram Of Clahe 16")
    plt.xlabel('Brightness')
    plt.ylabel('Pixels')
    

    plt.tight_layout()
    plt.show()
    


showAllEq(road1, road1Eq, road1_CL8,"clahe 8")
showAllEq(road2, road2Eq, road2_CL8,"clahe 8")
showAllEq(road3, road3Eq, road3_CL8,"clahe 8")


showAllEq(road1, road1Eq, road1_CL16,"clahe 16")
showAllEq(road2, road2Eq, road2_CL16,"clahe 16")
showAllEq(road3, road3Eq, road3_CL16,"clahe 16")

compareHist (road1 ,road1Eq, road1_CL8, road1_CL16)


# In[30]:


#3.1

img2 = cv2.imread('clock.jpg')
img2rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2gr = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(4,4))
plt.imshow(img2rgb)
plt.title("clock.jpg - Αρχική")
plt.axis('off')
plt.show()

#Μετέτρεψα την εικόνα σε float με ευρος 0,1 για τους υπολογισμούς του Sobel
imgGray = img2gr.astype(np.float32) / 255.0

#Υπολογισα κατα X (οριζόντια) και Y (καθετα) χρησιμοποιωντας φίλτρα Sobel.
sobelX = filters.sobel_h(imgGray)   
sobelY = filters.sobel_v(imgGray)  
#Υπολόγισα το μετρο της κλισης συνδυάζοντας τα X και Y.
sobel = np.hypot(sobelX, sobelY) 


plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(img2gr, cmap='gray')
plt.title("Grey")
plt.axis('off')

plt.subplot(2,2,2) 
plt.imshow(sobelX, cmap='gray')  
plt.title("Sobel X")   
plt.axis('off')

plt.subplot(2,2,3) 
plt.imshow(sobelY, cmap='gray')
plt.title("Sobel Y")
plt.axis('off')



plt.subplot(2,2,4) 
plt.imshow(sobel, cmap='gray')
plt.title("Sobel")
plt.axis('off')

plt.tight_layout()
plt.show()

#Εφαρμοσα τον αλγόριθμο Canny για ανιχνευση ακμων.
imgCanny = cv2.Canny(img2gr, threshold1=50, threshold2=150)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(imgGray, cmap='gray');     
plt.title("Grey") 
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(imgCanny, cmap='gray')  
plt.title("Canny")
plt.axis('off')

plt.tight_layout()
plt.show()

