import cv2
import sys
from predict import Detect_VP
import time
from deg_error import Draw_AA


#이미지 크기는 parameterization의 파일명에서 ht파일명의 앞에서부터를 기준으로 h,w를 나타낸다 여기서 *2의 값으로 넣어주면 된다.
#여기에 프레임별로 영상가져오면 label.txt로 점 찍고, predict 점 찍은것을 저장하고,
#predict된 값도 txt로 저장하기
def Get_Video_Frame(dir, yaml_name, model_path, save_dir):
    DELAY_FRAME = 381
    vidcap = cv2.VideoCapture(dir)
    detect = Detect_VP(model_path,yaml_name)
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(save_dir+"output.avi", fourcc, fps, (1280,720))
    txt_file = open(save_dir+"result.txt", "w")
    graph_file = open(save_dir+"result_graph.txt", "w")
    perform_txt_dir = open(save_dir+"perfomance.txt", "w")
    perform_avg = []


    while(vidcap.isOpened()):
        ret, image = vidcap.read()

        if(ret==False):
            print("\nvideo end")
            break

        if vidcap.get(cv2.CAP_PROP_POS_FRAMES) >= DELAY_FRAME:
            start_time = time.time()
            x,y,result_img,result_coord = detect.predict(image,(512,512), [10, 40, 80], int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))) #NYU 640,480 Scannet 512,512
            if x != -1:
                cv2.putText(result_img,f"x = {round(x, 2)} y = {round(y, 2)}",(10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            else:
                cv2.putText(result_img,f"not detected",(10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            cv2.putText(result_img,f"frame : {str(int(vidcap.get(cv2.CAP_PROP_POS_FRAMES)))}",(10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            out.write(result_img)
            end_time = time.time()
            coord_str = ""
            for coord in result_coord:
                coord_str += f",{round(coord[0],2)},{round(coord[1],2)}"
            txt_file.write(f"{int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))}{coord_str}\n")
            graph_file.write(f"{int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))},{round(x, 2)},{round(y, 2)}\n")
            
            perform_avg.append(end_time - start_time)
            perform_txt_dir.write(f"{str(int(vidcap.get(cv2.CAP_PROP_POS_FRAMES)))} {end_time - start_time:.5f} sec\n")
            sys.stdout.write("\rprogressed : "+str(int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))))
            sys.stdout.flush()

    AVG = sum(perform_avg) / len(perform_avg)
    perform_txt_dir.write(f"average : {AVG} sec.")
    txt_file.close()
    graph_file.close()
    perform_txt_dir.close()
    vidcap.release()
    Draw_AA.draw_graph(gt_dir = "etri_cart_200219_15h01m_2fps_gt3.txt",
                        rt_dir = "./saved_results/ScanNet/result_graph.txt",
                        save_dir = "./saved_results/ScanNet/AA_graph.png")


if __name__ == "__main__":
    Get_Video_Frame(dir = "/home/ubuntu/Desktop/etri_data/etri_cart_200219_15h01m_2fps.avi",
                    yaml_name = 'scannet',
                    model_path= "./model/scannet/checkpoint_latest.pth.tar",
                    save_dir = "./saved_results/ScanNet/")