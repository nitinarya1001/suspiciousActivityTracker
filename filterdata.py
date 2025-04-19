import os
import cv2
import numpy as np

result = 'frames'
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
for i in os.listdir(result):
    for j in os.listdir(os.path.join(result,i)):
        cv2.startWindowThread()
        img1 = cv2.imread(os.path.join(result,i,j))
        boxes, weights = hog.detectMultiScale(img1, winStride=(4, 4),padding=(4, 4),scale=1.05 )

        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        if boxes.shape[0]==0:
            os.remove(os.path.join(result,i,j))
            print("deleted file from: ",os.path.join(result,i,j))
        else:
            print("nothing is deleted: ",os.path.join(result,i,j))
        cv2.destroyAllWindows()
        # for (xA, yA, xB, yB) in boxes:  
        # # display the detected boxes in the colour picture
        #     cv2.rectangle(img1, (xA, yA), (xB, yB),(0, 255, 0), 2)
        # print(os.path.join(result,i,j))
        # cv2.imshow("bruh",img1)
        # cv2.waitKey(0)
    