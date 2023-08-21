import cv2
import sys
from eval import Detect_VP
import time
from deg_error import Draw_AA


def Get_Video_Frame(dir, model_path, save_dir):
    DELAY_FRAME = 381
    vidcap = cv2.VideoCapture(dir)
    detect = Detect_VP(model_path,"tmm17")
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
            x,y,result_img,result_coord = detect.predict(image,(512,512)) 
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
            txt_file.write(f"{int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))},{len(result_coord)}{coord_str}\n")
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
                        rt_dirs = ["./saved_results/tmm17/result_graph.txt"],
                        save_dir = "./saved_results/tmm17/AA_graph.png",
                        data_names=["tmm17"])


if __name__ == "__main__":
    Get_Video_Frame(dir = "/home/ubuntu/Desktop/etri_data/etri_cart_200219_15h01m_2fps.avi",
                    model_path= "./model/tmm17/checkpoint_best.pth.tar",
                    save_dir = "./saved_results/tmm17/")