import cv2
import matplotlib.pyplot as plt

array=[]
def click(event, x, y, flags, param):
       if event == cv2.EVENT_LBUTTONDOWN:
            print x,y
            array.append((x,y))
      
    
img=cv2.imread("image_gray.jpg")
cv2.namedWindow('image')
cv2.imshow('image',img)


while(1):
    #cv2.namedWindow('image')
    cv2.setMouseCallback('image',click)
    if cv2.waitKey(20) & 0xFF == 27:
        break
    
print array
cv2.destroyAllWindows()
