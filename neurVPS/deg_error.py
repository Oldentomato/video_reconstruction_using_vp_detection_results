from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import numpy as np
import math


class Draw_AA():

    @staticmethod
    def _AA(a, b, thresh):
        v = 0
        k = [i for i, val in enumerate(a) if val < thresh]

        if not k:
            return v

        a = [a[i] for i in k]
        a.append(thresh)

        n = len(a)
        for i in range(n - 1):
            v = v + (a[i + 1] - a[i]) * b[i]

        return v
    
    def _sync_data(gt_arr, rt_arr):

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

    @classmethod
    def draw_graph(cls, gt_dir, rt_dirs, save_dir, data_names):

        color = [[0.4660, 0.6740, 0.1880],[0.8500, 0.3250, 0.0980],[0.0980, 0.8500, 0.3250]]
        line_type = ['-','--']

        def filter(x):
            return np.where(x == -1, 0, x)
        

        # Reading data files
        gt = np.loadtxt(gt_dir, delimiter=',')  # index, vpcount, vx1, vy1
        for pr_index, rts in enumerate(rt_dirs):
            for index,rt_dir in enumerate(rts):
                rt = np.loadtxt(rt_dir, delimiter=',')  # index, vx1, vy1

                rt = cls._sync_data(gt, rt)

                f_arr = np.full((len(gt[:]),1),720)

                gt = filter(gt)
                rt = filter(rt)

                # f = gt[:, 2] / 2
                f = f_arr[:, 0] / 2 / math.tan(math.radians(82.1) / 2)


                a = np.column_stack((gt[:, 2:4], f))
                b = np.column_stack((rt[:, 1:3], f))
                n = len(f)

                err = np.zeros(n)
                for k in range(n):
                    v1 = a[k, :]
                    v2 = b[k, :]
                    err[k] = math.degrees(math.acos(np.dot(v2, v1) / (np.linalg.norm(v2) * np.linalg.norm(v1))))
                
                err = np.sort(err)
                pr = np.zeros(n)
                for k in range(n):
                    pr[k] = np.sum(err <= err[k]) / n

                a5 = cls._AA(err, pr, 5) / 5
                a10 = cls._AA(err, pr, 10) / 10
                a20 = cls._AA(err, pr, 20) / 20

                print(f"a5 = {a5}")
                print(f"a10 = {a10}")
                print(f"a20 = {a20}")

                # Plotting the graph

                # plt.plot(err2, pr2, color=[0.8500, 0.3250, 0.0980], linewidth=2)
                plt.plot(err, pr, line_type[pr_index], color=(np.random.random(), np.random.random(), np.random.random()), linewidth=2)


        plt.legend(data_names)
        plt.xlabel('Angle difference (degree)')
        plt.ylabel('Percentage')
        plt.xlim([0, 20])
        plt.ylim([0.4, 1])
        plt.grid(True)
        plt.savefig(save_dir)

