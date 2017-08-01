
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import greycomatrix, greycoprops
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# In[2]:

i=0
array=[]
array1=[]

def click1(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
	    ix,iy=x,y
	    font = cv2.FONT_HERSHEY_SIMPLEX
            s=""
	    s=s+str(ix)+","+str(iy)
	    cv2.putText(img,s,(ix,iy), font, 0.5,(0,0,0),2)
	    cv2.imshow('image',img)
            value=[(x,y)]
            array1.append(value)
	    


def click(event, x, y, flags, param):
    global i
    if event == cv2.EVENT_LBUTTONDOWN:
	    ix,iy=x,y
	    font = cv2.FONT_HERSHEY_SIMPLEX
            s=""
	    s=s+str(ix)+","+str(iy)
	    cv2.putText(img,s,(ix,iy), font, 0.5,(0,0,0),2)
	    cv2.imshow('image',img)
            value=[(x,y),i/15]
            i=i+1
            array.append(value)
	  
      
    
img=cv2.imread("image_gray.jpg",0)
cv2.namedWindow('image')
cv2.imshow('image',img)


while(1):
    #cv2.namedWindow('image')
    cv2.setMouseCallback('image',click)
    if  cv2.waitKey(0) & 0xFF == 27:
        break
    
print array
cv2.destroyAllWindows()


# glcm loops
# click window open for test
# data frames
# no csv
# trainig function
# traintestsplit()

# In[2]:

PATCH_SIZE = 20
x=[]
y=[]
xs=[]
image = cv2.imread("image_gray.jpg",0)
#plt.imshow(image ,cmap='gray')        
#plt.show()   


# In[ ]:

def feature(img):
    xs=[]
    glcm = greycomatrix(img, [5], [0], 256, symmetric=True, normed=True)
    xs.append(greycoprops(glcm, 'contrast')[0,0])
    xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    xs.append(greycoprops(glcm, 'homogeneity')[0, 0])
    xs.append(greycoprops(glcm, 'ASM')[0, 0])
    xs.append(greycoprops(glcm, 'energy')[0, 0])
    xs.append(greycoprops(glcm, 'correlation')[0, 0])
    return xs

for loc in array :
    patches=(image[loc[0][1]:loc[0][1] + PATCH_SIZE,loc[0][0]:loc[0][0] + PATCH_SIZE])
    x.append(feature(patches))
    y.append(loc[1])


# In[ ]:

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[11]:

knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn.fit(X_train,y_train)

scores = knn.score(X_test, y_test, sample_weight=None)
print "score ",scores


img=cv2.imread("image_gray.jpg",0)
cv2.namedWindow('image')
cv2.imshow('image',img)

while(1):
    cv2.setMouseCallback('image',click1)
    if  cv2.waitKey(0) & 0xFF == 27:
        break
    
print array1
cv2.destroyAllWindows()

for loc in array1:
    patches=(image[loc[0][1]:loc[0][1] + PATCH_SIZE,loc[0][0]:loc[0][0] + PATCH_SIZE])
    xs.append(feature(patches))
    
print xs

b=knn.predict(xs)
print b


#a=""



for i in b:
	if i in [0]:
		a="sky"
		img1=cv2.imread("image_gray.jpg",0)
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img1,a,(array1[0][0][0],array1[0][0][1]), font, 1,(0,0,0),2)
	        cv2.imshow('image',img1)
		cv2.waitKey(0)
		print a
	if i in [1]:
		a="grass"
		img2=cv2.imread("image_gray.jpg",0)		
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img2,a,(array1[0][0][0],array1[0][0][1]), font, 1,(0,0,0),2)
	        cv2.imshow('image',img2)
		cv2.waitKey(0)
		print a
			
	



