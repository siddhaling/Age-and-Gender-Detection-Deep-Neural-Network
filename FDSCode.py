pip install opencv-python

import cv2
import os
os.chdir('.../FDS PROJECT/nets')


def find(network,photo,limit=0.7):
    InputC=photo.copy()
    print(InputC.shape)
    length=InputC.shape[0]
    breadth=InputC.shape[1]
    dot=cv2.dnn.blobFromImage(InputC,1.0,(227,227),[124.96,115.97,106.13],swapRB=True,crop=False)
    network.setInput(dot)
    spots=network.forward()
    Markings=[]
    for l in range(spots.shape[2]):
        positive=spots[0,0,l,2]
        if positive>limit:
            a=int(spots[0,0,l,3]*breadth)
            b=int(spots[0,0,l,4]*length)
            m=int(spots[0,0,l,5]*breadth)
            n=int(spots[0,0,l,6]*length)
            Markings.append([a,b,m,n])
            cv2.rectangle(InputC,(a,b),(m,n),(255,182,193),int(round(length/150)),8)
    return InputC,Markings

fP='opencv_face_detector.pbtxt'
fM='opencv_face_detector_uint8.pb'
aP='age_deploy.prototxt'
aM='age_net.caffemodel'
gP='gender_deploy.prototxt'
gM='gender_net.caffemodel'

gL=['Male','Female']
aL=['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']
DetectNet=cv2.dnn.readNet(fM,fP)
ANet=cv2.dnn.readNet(aM,aP)
GNet=cv2.dnn.readNet(gM,gP)

#clip = cv2.imread('.../FDA PROJECT/AC_ECOM_SITE_2020_REFRESH_1_INDEX_M2_THUMBS-V2-1.jpg') #to read photo
clip=cv2.VideoCapture(0)	#to read video
space=20
while cv2.waitKey(1)<0:
    present,photo=clip.read() #to read video
    if not present:		#to read video
        cv2.waitKey()		#to read video
        break			#to read video
    #photo=clip	#to read photo
    img,Markings=find(DetectNet,photo)
    if not Markings:
        print("Face not found")
    for mark in Markings:
        face=photo[max(0,mark[1]-space):min(mark[3]+space,photo.shape[0]-1),max(0,mark[0]-space):min(mark[2]+space, photo.shape[1]-1)]
        dot=cv2.dnn.blobFromImage(face,1.0,(227,227),[124.96,115.97,106.13],swapRB=True,crop=False)
        GNet.setInput(dot)
        GPredict=GNet.forward()
        sex=gL[GPredict[0].argmax()]
        ANet.setInput(dot)
        APredict=ANet.forward()
        year=aL[APredict[0].argmax()]
        cv2.putText(img,f'{sex},{year}',(mark[0],mark[1]-10),cv2.FONT_ITALIC,0.8,(148, 0, 211),2,cv2.LINE_8)
        cv2.imshow("Detection of gender and age",img)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()