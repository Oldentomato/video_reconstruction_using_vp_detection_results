from deg_error import Draw_AA


Draw_AA.draw_graph(gt_dir = "etri_cart_200219_15h01m_2fps_gt3.txt",
                    rt_dirs = [["./saved_results/su3/result_graph.txt",
                               "./saved_results/scannet/result_graph.txt",
                               "./saved_results/tmm17/result_graph.txt"],
                               [
                                   "/home/ubuntu/Desktop/compare_result/cvpr22/saved_results/NYU/result_graph.txt",
                                   "/home/ubuntu/Desktop/compare_result/cvpr22/saved_results/SU3/result_graph.txt",
                                   "/home/ubuntu/Desktop/compare_result/cvpr22/saved_results/ScanNet/result_graph.txt"
                               ]],
                    save_dir = "./saved_results/AA_graph.png",
                    data_names=["SU3(neur)","ScanNet(neur)","tmm17(neur)"
                                ,"NYU(cvpr)","SU3(cvpr)","ScanNet(cvpr)"])