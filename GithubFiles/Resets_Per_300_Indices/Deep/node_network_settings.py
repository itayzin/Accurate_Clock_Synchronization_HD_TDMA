import torch

PERFORMANCE_PLOT = False
S_LENGTH = 30000
Data_ACQ_Time = 3000
DPL = 3010
progressive_period_update = True
EPS_0 = 0.3
PERIOD_EPS_0 = 0.3
EPOCHS = 60
SEED_ID = 643171342
N = 16
LOSS_TH = 0.9
Speed_Of_Light = 3e8
DITHER_STD = 0
TRAINING_DITHER_PERC = 1 / 100
RANDOMIZE_INITS = True
RANDOMIZE_LOCATIONS = True
EPOCHPLOT = False
DITHER_TRAINING_STD = 0#1e-3
NOISEPWR = 0#10e-7
TNOM = 1 / 200
DISCONNECTED = 0
BIAS_CORRECTION_ENABLED = False
links_l0 = 0.29
links_l1 = 0.31
amount_of_cores = 6
PRINT_ENABLED = False
alpha_weight_minmax_deep = 1 # 1 for complete deep, 0 for complete minmax
PPM_VAR = 150
Initial_Time_Scaling = 1
Jumps = 1
ind_shift = 0
network_reset_index = 12000 + N*Jumps + ind_shift
# reset_interval      = 3 * N * 30
reset_interval      = N * 270
reset_nodes_pick_ratio = 0.3

LEARNER_LIST = ["Phase", "Period"]
PHASE_LR = 0.1
PERIOD_LR = 0.1# Was 0.05
LOAD_SAFEDICT = True
CONTINUE_ENABLED = False
DLI = 3 # Deep Learning Iterations
pygamelocations_enabled = False
load_loc_from_file = False
NULL_PHASE_EPS_0 = 0
NULL_PERIOD_EPS_0 = 0

def loss_phase_element(element, k, v =torch.tensor(1)):
	return torch.abs(element)**2 * torch.log((k + v))
	pass
def loss_period_element(element, k, v =torch.tensor(1)):
	return torch.abs(element)**2 * torch.log((k + v))
	pass