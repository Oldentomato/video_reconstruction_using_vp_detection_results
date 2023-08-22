import cv2
import sys
import numpy as np


def sync_data(gt_arr, rt_arr):

    new_rt_arr = []
    pred_frame_count = 0

    for frame_count in range(0,len(gt_arr[:,0])):
        if int(rt_arr[pred_frame_count, 0]) == int(gt_arr[frame_count, 0]):
            new_rt_arr.append(rt_arr[pred_frame_count])
            pred_frame_count += 1
        else:
            pred_frame_count = int(gt_arr[frame_count, 0]) - int(gt_arr[frame_count-1, 0]) + pred_frame_count
            new_rt_arr.append(rt_arr[pred_frame_count])

    return np.array(new_rt_arr)

def Get_Video_Frame(dir,gt_dir, path_neur, path_cvpr, save_dir):
    DELAY_FRAME = 381
    vidcap = cv2.VideoCapture(dir)
    gt = np.loadtxt(gt_dir, delimiter=',')
    neur = np.loadtxt(path_neur, delimiter=',')
    cvpr = np.loadtxt(path_cvpr, delimiter=',')

    neur = sync_data(gt, neur)
    cvpr = sync_data(gt, cvpr)


    now_frame = 0
    index = 0

    fps = vidcap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(save_dir+"output.avi", fourcc, fps, (1280,720))

    while(vidcap.isOpened()):
        ret, image = vidcap.read()

        if(now_frame>=len(gt)-1):
            print("\nvideo end")
            break

        now_frame = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))

        if now_frame >= DELAY_FRAME:
            if gt[index,0] == now_frame:
                result_img = cv2.circle(image, (int(neur[index,1]),int(neur[index,2])), radius=20, color=(255,0,0), thickness=-1)
                result_img = cv2.circle(result_img, (int(cvpr[index,1]),int(cvpr[index,2])), radius=20, color=(0,255,0), thickness=-1)
                result_img = cv2.circle(result_img, (int(gt[index,2]),int(gt[index,3])), radius=20, color=(0,255,255), thickness=-1)
                if neur[now_frame,1] != -1:
                    cv2.putText(result_img,f"neur(su3) x = {round(neur[now_frame,1], 2)} y = {round(neur[now_frame,2], 2)}",(10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
                else:
                    cv2.putText(result_img,f"not detected",(10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
                if cvpr[now_frame,1] != -1:
                    cv2.putText(result_img,f"cvpr(nyu) x = {round(cvpr[now_frame,1], 2)} y = {round(cvpr[now_frame,2], 2)}",(10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                else:
                    cv2.putText(result_img,f"not detected",(10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                cv2.putText(result_img,f"frame : {str(now_frame)}",(10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,50,127), 2)
                out.write(result_img)
            else:
                continue
            index += 1
            sys.stdout.write("\rprogressed : "+str(now_frame))
            sys.stdout.flush()

    vidcap.release()



if __name__ == "__main__":
    Get_Video_Frame(dir = "/home/ubuntu/Desktop/etri_data/etri_cart_200219_15h01m_2fps.avi",
                    gt_dir="./neurVPS/etri_cart_200219_15h01m_2fps_gt3.txt",
                    path_neur= "./neurVPS/saved_results/su3/result_graph.txt",
                    path_cvpr= "./cvpr22/saved_results/NYU/result_graph.txt",
                    save_dir = "./")