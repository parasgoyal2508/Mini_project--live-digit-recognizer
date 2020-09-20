import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
m_new = tf.keras.models.load_model('mnist_dataset.h5')

img = np.ones([400,400],dtype ='uint8')*255
img[50:350,50:350]=0
wname = 'Canvas'
cv2.namedWindow(wname)
drawing = False 

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        
    if event == cv2.EVENT_LBUTTONUP:
        drawing = False
        
    if event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
             cv2.circle(img,(x,y),10,(255,0,0),-1)
             print(x,y)

#call back function
cv2.setMouseCallback(wname,draw_circle)


while True:
    cv2.imshow(wname,img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        img[50:350,50:350]=0
    elif key == ord('w'):
        out = img[50:350,50:350]
        cv2.imwrite('Output.jpg',out)
    elif key == ord('g'):
        image_test = img[50:350,50:350]
        image_test_resize = cv2.resize(image_test,(28,28)).reshape(1,28,28)
        z=m_new.predict_classes(image_test_resize)
        print(z)
cv2.destroyAllWindows()
