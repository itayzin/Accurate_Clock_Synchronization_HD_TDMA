from pygame_location_gen import InteractiveFigure
import subprocess
import time
from datetime import timedelta
import numpy as np

Game = InteractiveFigure()
locations = np.array(Game.Locations) * 10
np.savetxt(f"Locations_{nns.SEED_ID}.npy", locations)
def increment_seed():
    with open("node_network_settings.py", "r") as f:
        lines = f.readlines()
    for i in range(len(lines)):
        if "SEED_ID" in lines[i]:
            bines = lines[i].replace("SEED_ID = ", "");bines = int((bines.split(")")[0]));lines[i] = f"SEED_ID = {bines + 1000}\n"
    with open("node_network_settings.py", "w") as f:
        f.writelines(lines)
    pass

def Prog_Time_Est(k, k_max, total_time):
    prog = k / k_max
    print(f"Progress: {prog * 100:.4f}%")
    iters_left = k_max - k
    est_time = total_time * iters_left / k
    print(f"Estimated Time remaining: {str(timedelta(seconds=est_time))}")


if __name__ == '__main__':
	elapsed_time_vec = []

	MA = 0
	i_start = 1
	i_end = 800
	MA_N = 10
	print(f"Started running {i_end} Simulations!")
	for i in range (i_start, i_end):
		start_time = time.time()

		subprocess.call(["python", 'PY_NET_COMBINED-3.py'])
		subprocess.call(["python", 'py_net_no_learning.py'])

		increment_seed()
		end_time = time.time()

		elapsed_time = end_time - start_time
		elapsed_time_vec.append(elapsed_time)
		MA = np.sum(elapsed_time_vec[-MA_N:-1]) / MA_N
		Prog_Time_Est(i, i_end, MA)
