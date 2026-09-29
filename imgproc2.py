#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np
import matplotlib.pyplot as plt
import cv2 
import scipy.io as sio
from scipy.fftpack import dct, idct
from scipy.signal import wiener, convolve2d


def norm_img(img) :

    
#1.Προεπεξεργασία (Γραμμική μετατόπιση τιμών στο [0, 255])

     img_min =  np.min(img)
     img_max =  np.max(img)
     #Τύπος κανονικοποίησης που χρησιμοποίησα : (pixel - min)/(max - min)*255
     normalized =  255 * (img - img_min ) /(img_max - img_min)
     return normalized

 #2. Δημιουργία πίνακα για το κεντράρισμα συχνότητας. Πολλαπλασιαζω με (-1)^(x+y).
def get_cent( shape ):
    
    rows, cols = shape
    x = np.arange(rows)
    y = np.arange(cols)
    #Δημιουργία grid συντεταγμένων
    X , Y = np.meshgrid( x, y, indexing='ij' ) 

    
    # Φτιαχνω τον πίνακα (-1)^(x+y) όπου μετακινεί τη συχνότητα 0,0 στο κέντρο της εικόνας
    cent_matr = np.power(-1, X + Y )
    return cent_matr

# 3. Υπολογιζω DFT με χρήση μόνο της fft. Εφαρμόζω fft πρωτα στις γραμμές και μετά στις στήλες.
def  calc_dft( img ) :
    
    #FFT κατά μήκος των γραμμών (axis=0)
    fft_rows = np.fft.fft(img, axis=0)

    
    # FFT κατά μήκος των στηλων (axis=1)
    dft_result = np.fft.fft(fft_rows, axis=1)

    
    return dft_result



    # 5 Αντίστροφη μετατροπή (IDFT). Εφαρμόζω ifft πρώτα στις γραμμές, μετά στις στήλες.
def calc_idft(img_freq):
   
     #IFFT κατά μήκος των γραμμών
    ifft_rows = np.fft.ifft(img_freq, axis=0)
    
    # IFFT κατά μήκος των στήλών
    idft_result = np.fft.ifft(ifft_rows, axis=1)
    
    return idft_result



    #4. Δημιουργία φίλτρου Notch (μάσκα με 0 στα σημεία αποκοπής και 1 αλλού).
def notch_filt(shape, notch_points, radius=5):
   
    rows, cols = shape
    mask = np.ones((rows, cols), dtype=np.float32)
    
    center_x, center_y = rows // 2, cols // 2
    
    y, x = np.ogrid[:rows, :cols]
    
    for u, v in notch_points:
        # Δημιουργούμε "κύκλο" με τιμές 0 γύρω από το σημείο θορύβου
        # Πρέπει να κόψουμε και το συμμετρικό του σημείο ως προς το κέντρο
        
        # Απόσταση από το σημείο (u, v)
        dist1 = (x - v)**2 + (y - u)**2
        mask[dist1 <= radius**2] = 0
        
        # Συμμετρικό σημείο
        sym_u = 2*center_x - u
        sym_v = 2*center_y - v
        dist2 = (x - sym_v)**2 + (y - sym_u)**2
        mask[dist2 <= radius**2] = 0
        
    return mask




mat_contents = sio.loadmat("lenna.mat")

# Αποθηκέυεται η εικόνα στην μεταβλητ΄΄η img εκτός από τα σκουπίδια (δηλαδή οτι ξεκινάει απ΄΄ο _)
key = [k for k in mat_contents.keys() if not k.startswith('_')][0]
img = mat_contents[key]


# Μετατροπή σε float για πράξεις
img = img.astype(float)

 # 1. Προεπεξεργασία (Normalization)
img_normalized = norm_img(img)

# 2. Κεντράρισμα Συχνοτήτων 
# Πολλαπλασιάζω την εικόνα με (-1)^(x+y)
cent_matr = get_cent(img_normalized.shape)
img_centered = img_normalized * cent_matr

# 3. Υπολογίζω DFT (Χρήση fft σε γραμμ΄΄ες και σε στήλες) 
dft_centered = calc_dft(img_centered)

#Οπτικοποίηση φάσματος 
# Το φάσμα θα μου δείξει πόσο 'δυνατή' είναι κάθε συχνότητα.
# Χρησιμοποίησα log για να φανούν οι λεπτομέρειες γιατί είχα μεγάλες διαφορές
spectrum = np.log(1 + np.abs(dft_centered))

 # 4. Φιλτράρισμα Notch 
#Έιδα οτι η εκφώνηση λέει δοσμένο φίλτρο αλλά δεν είδα συντεταγμένες,
#οποτε όρισα ενδεικτικά σημεία θορύβου.
notch_coordinates = [
    (100, 100), # Παράδειγμα 1
    (50, 200) ]  # Παράδειγμα 2 

notch_mask = notch_filt(img.shape, notch_coordinates, radius=10)

 # Εφαρμ΄΄ώζω το φίλτρο πολλαπλασιάζοντας στοιχείο προς στοιχείο 
dft_filtered = dft_centered * notch_mask

#5 Αντίστροφη μετατροπή (IDFT)
idft_centered = calc_idft(dft_filtered)

 # Πήρα το πραγματικό μέρος (real) γιατί το αποτέλεσμα  έχει αμελητέο φανταστικό μέρος
img_reconstructed_centered = np.real(idft_centered)

 #6 . Αναμετατόπιση σημεiου στο (0,0) 
 # Πολλαπλασιάζω ξανά με (-1)^(x+y) για να φύγει το κε΄ντράρισμα
img_final = img_reconstructed_centered * cent_matr


# Εμφάνιση αποτελέσματος
plt.figure(figsize=(12, 10))

# Αρχική εικόνα
plt.subplot(2, 3, 1)
plt.imshow(img_normalized, cmap='gray')
plt.title(" 1.Αρχική ")
plt.axis('off')

#Φάσμα συχνοτήτων
plt.subplot(2, 3, 2)
plt.imshow(spectrum, cmap='gray')
plt.title("3.Φάσμα (Κεντραρισμένο)")
plt.axis('off')

 # Φίλτρο Notch
plt.subplot(2, 3, 3)
plt.imshow(notch_mask, cmap='gray')
plt.title("4. Notch φίλτρο ")
plt.axis('off')

 # Φιλτραρισμένο Φάσμα
plt.subplot(2, 3, 4)
plt.imshow(np.log(1 + np.abs(dft_filtered)), cmap='gray')
plt.title("Φιλτραρισμένο φάσμα")
plt.axis('off')

 #Τελική Εικονα
plt.subplot(2, 3, 5)
plt.imshow(img_final, cmap='gray')
plt.title(" 6. Τελική (Αναμετατόπιση)")
plt.axis('off')

plt.tight_layout()
plt.show()


# In[53]:


mat = sio.loadmat('tiger.mat')
key = [k for k in mat.keys() if not k.startswith('__')][0]
img = mat[key]


# Μετατροπή σε float (για τις πράξεις) και κανονικοποίηση
img = img.astype(float)
M, N = img.shape



# In[54]:


def dct2(a):
    # Εφαρμογή DCT στις γραμμές και μετά στις στήλες
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    # Αντίστροφος DCT
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

def calc_mse(original, compressed):
    # Υπολογισμός Μέσου Τετραγωνικού Σφάλματος  (MSE)
    orig_float = original.astype("float")
    comp_float = compressed.astype("float")
    err = np.sum((orig_float - comp_float) ** 2)
    err /= float(original.shape[0] * original.shape[1])
    return err


# In[55]:


def get_zigzag_mask(n, keep_count):
    # Δημιουργία μάσκας Zigzag 
    mask = np.zeros((n, n), dtype=int)
    rows, cols = n, n
    
    # Πίνακας με τη σειρά zigzag 
    index_order = np.zeros((n, n), dtype=int)
    cur_row, cur_col = 0, 0
    cur_index = 0
    
    #Φτιάχνω μια λίστα με όλες τις θέσεις (γραμμή, στήλη) του πίνακα.
    #Αντί να γράψω δύσκολο κώδικα για κίνηση Zigzag, κάνω το κόλπο Manhattan Distance 
    # όπου το βρήκα στο https://www.statology.org/manhattan-distance-python/ 
    #Υπολογίζω το άθροισμα "γραμμή + στήλη" για κάθε κουτί.
   #Όσο πιο μικρό είναι το άθροισμα, τόσο πιο κοντά είναι το κουτάκι 
   #στην πάνω αριστερή γωνία (εκεί που είναι η χρήσιμη πληροφορία).
    
    indices = []
    for r in range(n):
        for c in range(n):
            # Αποθηκεύω άθροισμα, γραμμή, στήλη αντίστοιχα
            indices.append((r + c, r, c)) # Ταξινομώ με βάση το άθροισμα δεικτών (διαγώνια)

    # Ταξινόμηση από το μικρότερο στο μεγαλύτερο. Έτσι τα πρώτα στοιχεία
    #στην λίστα θα είναι αυτά που είναι πάνω αριστερά
    indices.sort() 
    

    # Κρατάω μόνο όσα χρειάζομαι και βάζω 1 στη μάσκα σε αυτές τις θέσεις
    for i in range(keep_count):
        _, r, c = indices[i]
        mask[r, c] = 1
        
    return mask

block_size = 32  # Οπως ζητείται στην εκφώνηση 
p_values = [0.05, 0.20, 0.35, 0.50]  # Ποσοστά p

 # Για αποθήκευση αποτελεσμάτων MSE
mse_threshold = []
mse_zigzag = []

#Εικόνες για εμφάνιση 
img_thresh_display = np.zeros_like(img)
img_zigzag_display = np.zeros_like(img)

print(f"\n Διαστάσεις εικόνας : {M},{N}")
print(f" Μέγεθος μπλοκ: {block_size},{block_size}")

#Βρόχος για κάθε ποσοστό p
for p in p_values:
    # Δημιουργώ κεν΄΄ες εικόνες για το αποτέλεσμα
    recon_thresh = np.zeros_like(img)
    recon_zigzag = np.zeros_like(img)
    
    # Πόσους συντελεστές θα κρατήσω 
    #Σύνολο συντελεστών = 32 * 32 = 1024
    coeffs_to_keep = int(p * block_size * block_size)
    
    # Υπολογισμός μάσκας Zigzag για αυτό το p
    mask_Z = get_zigzag_mask(block_size, coeffs_to_keep)
    
    # Σάρωση της εικόνας ανά μπλοκ
    for i in range(0, M, block_size):
        for j in range(0, N, block_size):
    
            # Στην περίπτωση που η εικόνα δεν διαιρείται ακριβώς :
            actual_h = min(block_size, M - i)
            actual_w = min(block_size, N - j)
            
            block = img[i:i+actual_h, j:j+actual_w]
            
            # Αν το μπλοκ δεν είναι τετράγωνο , το προσπερνω.
            if actual_h != block_size or actual_w != block_size:
                continue

            #Εφαρμογή 2D-DCT
            dct_block = dct2(block)
            
           #Κατώφλι (Threshold)
            flat = np.abs(dct_block).flatten()
            flat.sort() 
            # Βρίσκω την τιμή κατωφλίου
            thresh_val = flat[-coeffs_to_keep] 
            
           # Δημιουργία μάσκας (κρατάμε όσα είναι μεγα΄λύτερα η ίσα με το thresh_val)
            mask_T = np.abs(dct_block) >= thresh_val
            
            #Φιλτράρισμα και Αντίστροφος DCT
            dct_thresh = dct_block * mask_T
            recon_thresh[i:i+block_size, j:j+block_size] = idct2(dct_thresh)
            
            # Μέθοδος Ζώνης (Zigzag) 
            #Εδώ η μάσκα είναι σταθερή (mask_Z) και κρατάει τις χαμηλές συχνότητες
            dct_zig = dct_block * mask_Z
            recon_zigzag[i:i+block_size, j:j+block_size] = idct2(dct_zig)
            
 #  3. Υπολογιζω MSE για το τρέχον p
    err_t = calc_mse(img, recon_thresh)
    err_z = calc_mse(img, recon_zigzag)
    
    mse_threshold.append(err_t)
    mse_zigzag.append(err_z)
    
    print(f"P={p*100}% -> MSE Threshold: {err_t:.2f}, MSE Zigzag: {err_z:.2f}" )

    
    # Αποθήκευση για εμφάνιση (στο τελευταίο loop)
    img_thresh_display = recon_thresh
    img_zigzag_display = recon_zigzag



# MSE
plt.figure(figsize=(10, 5))
plt.plot([x*100 for x in p_values], mse_threshold, 'o-', label='Μέθοδος Threshold')
plt.plot([x*100 for x in p_values], mse_zigzag, 's-', label='Μέθοδος Zigzag')
plt.title('Σφάλμα MSE ανά Μέθοδο')
plt.xlabel('Ποσοστό Διατήρησης Συντελεστών')
plt.ylabel('MSE ')
plt.legend()
plt.grid(True)
plt.show()



plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Αρχική ')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_thresh_display, cmap='gray')
plt.title(f'Threshold μέθοδος , (p={p_values[-1]*100}%)')
plt.axis('off')


plt.subplot(1, 3, 3)
plt.imshow(img_zigzag_display, cmap='gray')
plt.title(f'Zigzag μέθοδος, (p={p_values[-1]*100}%)')
plt.axis('off')

plt.show()


# In[56]:


#3 - Αποκατάσταση Εικόνας (Παρόμοια με την 1η εργαστηριακή εργασία)

#Υπολογίζω το MSE
def calc_mse(original, processed):
    
    return np.mean(( original - processed )**2)


mat = sio.loadmat('olympics.mat')
key = [k for k in mat.keys() if not k.startswith('__')][0]
img =  mat[key]

# Κανονικοποίηση  (Normalization)
img =  img.astype(float)
img = ( img - np.min(img)) / (np.max(img) - np.min(img) )

rows, cols = img.shape

#Αφαίρεση Θορύβου (Noise Removal)

print("Αφαίρεση Θορύβου")

# Προσθήκη Gaussian Θορύβου 
SNR_dB = 10
 # Ισχύς σήματος 
P_signal = np.mean(img ** 2)

 #Υπολογισμός ισχύος θορύβου βάσει SNR
# SNR_dB = 10 * log10(P_signal / P_noise) => P_noise = P_signal / 10^(SNR/10)
P_noise =  P_signal / (10 ** (SNR_dB / 10))
sigma_noise = np.sqrt( P_noise )

# Δημιουργία θορύβου
noise = np.random.normal( 0, sigma_noise , (rows, cols) )
img_noisy = img + noise

#Αφαίρεση με φίλτρο Wiener 
# Επειδή εδώ έχουμε μόνο θόρυβο, χρησιμοποιω το Wiener στο χωρικό πεδίο.
img_denoised = wiener(img_noisy, (5, 5)) 

# Αποτελέσματα
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1); 
plt.imshow(img, cmap='gray'); 
plt.title('Αρχική (Original)')
plt.axis('off')


plt.subplot(1, 3, 2); 
plt.imshow(img_noisy, cmap='gray'); 
plt.title('Με Θόρυβο SNR=10dB')
plt.axis('off')
plt.subplot(1, 3, 3); 
plt.imshow(img_denoised, cmap='gray'); 
plt.title('Wiener ')
plt.axis('off')
plt.show()


# In[ ]:




