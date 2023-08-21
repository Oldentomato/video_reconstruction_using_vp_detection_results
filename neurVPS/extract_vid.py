import cv2
import sys

DELAY_FRAME = 381
vidcap = cv2.VideoCapture("/home/ubuntu/Desktop/etri_data/etri_cart_200219_15h01m_2fps.avi")
fps = vidcap.get(cv2.CAP_PROP_FPS)

while(vidcap.isOpened()):
    ret, image = vidcap.read()

    if(ret==False):
        print("\nvideo end")
        break

    if vidcap.get(cv2.CAP_PROP_POS_FRAMES) >= DELAY_FRAME:
        cv2.imwrite(f"./images/{int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))}.png",image)
        sys.stdout.write("\rprogressed : "+str(int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))))
        sys.stdout.flush()
