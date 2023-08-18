from deg_error import Draw_AA

Draw_AA.draw_graph(gt_dir = "etri_cart_200219_15h01m_2fps_gt3.txt",
                    rt_dirs = ["./saved_results/NYU/result_graph.txt","./saved_results/ScanNet/result_graph.txt","./saved_results/SU3/result_graph.txt"],
                    save_dir = "./saved_results/AA_graph.png",
                    data_names=["NYU","ScanNet","SU3"])
# a5 = 0.5294747685510703
# a10 = 0.631757515723608
# a20 = 0.6945659323426412
# a5 = 0.2617317781469741
# a10 = 0.34012258043360066
# a20 = 0.46955528087802784
# a5 = 0.5260258722559374
# a10 = 0.6032862898327569
# a20 = 0.6680646283928197