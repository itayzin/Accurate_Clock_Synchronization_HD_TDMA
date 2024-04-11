import logging
import concurrent.futures
import scipy as sp
import torch as torch
import matplotlib.pyplot as plt
import numpy as np
import os
from torch import nn
import networkx as nx
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RANSACRegressor, TheilSenRegressor, \
	HuberRegressor
from scipy.signal import savgol_filter
import sys
from node_for_deep_sim import node
import node_network_settings as nns
import json
import pickle
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import random
from copy import deepcopy

try:
	os.mkdir("pre_generated_setups")
except:
	pass

def multiformation_plot(Node_Aggregation, SIM_LEN, network_type="Random"):
	plt.figure()
	plt.subplot(221)
	Node_Aggregation.plot_phases(SIM_LEN - 100)
	plt.title(f"Phases of {network_type} Network")
	plt.legend([k for k in range(1, nns.N + 1)])

	plt.subplot(222)
	Node_Aggregation.plot_periods(SIM_LEN - 100)
	plt.title(f"Periods of {network_type} Network")
	plt.legend([k for k in range(1, nns.N + 1)])

	plt.subplot(223)
	Node_Aggregation.plot_periods_skew(SIM_LEN - 100)
	plt.title(f"Periods minus Mean Period of {network_type} Network")
	plt.legend([k for k in range(1, nns.N + 1)])

	plt.subplot(224)
	Node_Aggregation.plot_periods_average(SIM_LEN - 100)
	plt.title(f"Mean Period of {network_type} Network")
	plt.legend([k for k in range(1, nns.N + 1)])
	plt.show()


def F_lambda(fa, fb, N):
	return fa + (fb - fa) * np.random.rand(nns.N, 1)


def Temporary_Dictionary_Preparation(load_path=False):
	#     nns.S_LENGTH = 28e2
	load_emeka = False
	load_itay_matlab = False
	sim_properties_dict = {}
	y_max = 1e4
	x_max = 1e4
	nns.EPS_0 = nns.EPS_0
	period_eps_0 = nns.PERIOD_EPS_0  # 0.03# Need to be adjusted for frequency dispairities?
	T_Nom = nns.TNOM
	sim_properties_dict["N"] = nns.N
	sim_properties_dict["TX_Power"] = [
		2 for x in range(nns.N)]  # 33 dBm times 1.5^4
	sim_properties_dict["eps"] = [nns.EPS_0 for x in range(nns.N)]
	sim_properties_dict["period_eps"] = [period_eps_0 for x in range(nns.N)]
	sim_properties_dict["TNOM"] = T_Nom
	sim_properties_dict["Initial_Period"] = [0] * nns.N
	sim_properties_dict["Initial_Time"] = [0] * nns.N
	sim_properties_dict["location"] = [[0, 0]] * nns.N
	sim_properties_dict["x_max"] = 1e4
	sim_properties_dict["y_max"] = 1e4
	locx = 1e3 * np.array(
		[[9.9948, 9.0057], [0.4578, 6.5922], [9.4538, 1.7647], [9.8315, 9.1101], [2.6137, 9.5124], [3.5478, 8.3183],
		 [4.8136, 1.2489], [2.6271,
							9.2686], [2.7696, 1.1088], [9.6264, 4.3108], [9.6251, 8.3644], [5.0140, 7.1690],
		 [8.5213, 7.3747], [8.8435, 8.9857], [3.3729, 2.3853], [2.4847, 9.8258]])
	timx = [0.00373679954512025, 0.00168472306717067, 0.00370353279091433, 0.00222791377443637, 0.00182026384352094,
			0.00254556818757784, 0.00445734624062783, 0.00495488819672182,
			0.000761994193827854, 0.00456980314708069, 0.00105723661276741, 0.00358234467143657, 0.00319751306610202,
			0.00247622668886020, 0.00134521952997997, 0.00314523196004965]
	perx = [0.00500041556879574, 0.00500002917606793, 0.00499937478181991, 0.00500017703319659, 0.00500033205889684,
			0.00499930677387600, 0.00499949663776904, 0.00500021863467411,
			0.00499983297518691, 0.00499947382850121, 0.00499950425341721, 0.00500004842001567, 0.00500039899944921,
			0.00499988812924100, 0.00500014106248181, 0.00500054683409894]
	for ID in range(nns.N):
		B0 = float(1 - 2 * np.random.randint(0, 2))
		A = float(4 + 2 * np.random.rand(1))
		Ti = float(T_Nom * (1 + B0 * 10 ** (- A)))
		t0 = float(Ti * np.random.rand(1))
		#         sim_properties_dict["Initial_Period"][ID] = Ti
		sim_properties_dict["Initial_Time"][ID] = timx[ID]
		#         sim_properties_dict["Initial_Time"][ID] = t0
		sim_properties_dict["Initial_Period"][ID] = perx[ID]
		#         sim_properties_dict["location"][ID] = [float(np.random.rand(1) * x_max), float(np.random.rand(1) * y_max)]
		sim_properties_dict["location"][ID] = [
			float(locx[ID, 0]), float(locx[ID, 1])]
	sim_properties_dict["sim_length"] = int(nns.S_LENGTH)
	# sim_properties_dict["Data_ACQ_Time"] = (nns.N + 1) * nns.N + 1
	#     sim_properties_dict["Data_ACQ_Time"] = nns.N * 10
	sim_properties_dict["Data_ACQ_Time"] = nns.Data_ACQ_Time
	sim_properties_dict["Power_Threshold"] = 3.9810717055e-15
	sim_properties_dict["Power_Exponent"] = 4
	sim_properties_dict["Speed_Of_Light"] = nns.Speed_Of_Light
	return sim_properties_dict


def gen_alpha_N(connectivity_mat, per_flag):
	alpha_mat = np.zeros(connectivity_mat.shape)
	g = nx.from_numpy_array(connectivity_mat, create_using=nx.Graph)
	bins = list(nx.connected_components(g))
	fully_connected_flag = len(bins) == 1
	if fully_connected_flag:
		for i in range(alpha_mat.shape[0]):
			for j in range(alpha_mat.shape[1]):
				if connectivity_mat[i, j] == 1:
					if per_flag:
						alpha_mat[i, j] = 1 / np.sum(connectivity_mat[:, j])
					else:
						alpha_mat[i, j] = 1 / connectivity_mat.shape[1]
				else:
					alpha_mat[i, j] = 0
				if not fully_connected_flag:
					break
			if not fully_connected_flag:
				break
	return alpha_mat, fully_connected_flag


def gen_random_softmax_weights(connectivity_mat):
	alpha_mat = np.zeros(connectivity_mat.shape)
	g = nx.from_numpy_array(connectivity_mat)
	bins = list(nx.connected_components(g))
	fully_connected_flag = len(bins) == 1
	if fully_connected_flag:
		for i in range(alpha_mat.shape[0]):
			for j in range(alpha_mat.shape[1]):
				if connectivity_mat[i, j] == 1:
					alpha_mat[i, j] = abs(np.random.randn(1))
				else:
					alpha_mat[i, j] = -99999999
				if not fully_connected_flag:
					break
			if not fully_connected_flag:
				break
	alpha_mat = softmax(alpha_mat) * connectivity_mat
	return alpha_mat, fully_connected_flag


def get_q_alphas_connectivity(xy, power_exponent, power_threshold, p0, SOL):
	nns.N = xy.shape[0]
	con_graph = np.ones((nns.N, nns.N)) - np.eye(nns.N)
	distance_mat = calc_distances(xy)
	for i in range(nns.N):
		for j in range(nns.N):
			if con_graph[i, j] == 1:
				if p0 * distance_mat[i, j] ** (-power_exponent) < power_threshold:
					con_graph[i, j] = 0
					nns.DISCONNECTED += 1
	q_mat = calc_qs(distance_mat, SOL)
	alpha_mat = calc_alphas(con_graph, distance_mat, power_exponent, p0)
	return q_mat, alpha_mat, con_graph

def get_q_alphas_connectivity(xy, power_exponent, power_threshold, p0, SOL):
	nns.N = xy.shape[0]
	con_graph = np.ones((nns.N, nns.N)) - np.eye(nns.N)
	distance_mat = calc_distances(xy)
	for i in range(nns.N):
		for j in range(nns.N):
			if con_graph[i, j] == 1:
				if p0 * distance_mat[i, j] ** (-power_exponent) < power_threshold:
					con_graph[i, j] = 0
					nns.DISCONNECTED += 1
	q_mat = calc_qs(distance_mat, SOL)
	alpha_mat = calc_alphas(con_graph, distance_mat, power_exponent, p0)
	return q_mat, alpha_mat, con_graph

def calc_qs(distance_mat, SOL):
	nns.N = distance_mat.shape[1]
	qs_mat = np.zeros((nns.N, nns.N))
	c = SOL
	for i in range(nns.N):
		for j in range(nns.N):
			qs_mat[i, j] = distance_mat[i, j] / c
	return qs_mat


def calc_alphas(con_graph, distance_mat, power_exponent, p0):
	nns.N = con_graph.shape[1]
	alpha_mat = np.zeros((nns.N, nns.N))
	neigh_pow = np.zeros(nns.N)
	for i in range(nns.N):
		neigh_pow[i] = power_sum(
			i, con_graph, distance_mat, power_exponent, p0)
		for j in range(nns.N):
			if con_graph[i, j] == 1:
				alpha_mat[i, j] = p0 * distance_mat[i,
				j] ** (-power_exponent) / neigh_pow[i]
	return alpha_mat.T


def calc_distances(xy_mat):
	nns.N = xy_mat.shape[0]
	distance_mat = np.zeros((nns.N, nns.N))
	for i in range(nns.N):
		for j in range(nns.N):
			distance_mat[i, j] = np.sqrt(
				(xy_mat[i, 0] - xy_mat[j, 0]) ** 2 + (xy_mat[i, 1] - xy_mat[j, 1]) ** 2)
	return distance_mat


def power_sum(i, con_graph, distance_mat, power_exponent, p0):
	out_sum = 0
	nns.N = con_graph.shape[1]
	for k in range(nns.N):
		if con_graph[i, k] == 1:
			out_sum += p0 * distance_mat[i, k] ** (-power_exponent)
	return out_sum


def softmax(x):
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum(axis=0)


class SlowLearningScalarMultiplication(nn.Module):
	def __init__(self, initial_value=5.0, slow_learning_rate=0.01):
		super(SlowLearningScalarMultiplication, self).__init__()
		self.slowscaler = nn.Parameter(
			torch.tensor(initial_value), requires_grad=True)
		self.slow_learning_rate = slow_learning_rate

	def forward(self, x):
		return self.slowscaler * x


class Learning_Softmaxer(nn.Module):
	def __init__(self, initial_value=1.23, slow_learning_rate=1):
		super(Learning_Softmaxer, self).__init__()
		self.slowscaler = nn.Parameter(
			torch.tensor(initial_value), requires_grad=True)
		self.slow_learning_rate = slow_learning_rate

	def forward(self, x):
		return nn.functional.softmax(nns.N * x[:nns.N])




def reset_non_connected(input, connection_vector):
	for i in range(len(connection_vector) // 2):
		if connection_vector[i] == 0:
			with torch.no_grad():
				input[i] = 0
				input[i + len(connection_vector) // 2] = 0

	return input


def reset_non_connected_single(input, connection_vector):
	for i in range(len(connection_vector)):
		if connection_vector[i] == 0:
			with torch.no_grad():
				input[i] = 0

	return input


class node_aggregation():
	def __init__(self, sim_properties_dict):
		self.node_array = create_node_array(sim_properties_dict)
		self.safedict = sim_properties_dict
		self.Power_Threshold = sim_properties_dict["Power_Threshold"]
		self.Power_Exponent = sim_properties_dict["Power_Exponent"]
		self.dist_mat = torch.zeros(
			sim_properties_dict["N"], sim_properties_dict["N"])
		self.power_mat = torch.zeros(
			sim_properties_dict["N"], sim_properties_dict["N"])
		self.connection_mat = torch.zeros(
			sim_properties_dict["N"], sim_properties_dict["N"])
		self.prop_delay_mat = torch.zeros(
			sim_properties_dict["N"], sim_properties_dict["N"])

	def Assemble_Properties(self):
		for i in range(len(self.node_array)):
			# row -> reception i here means received power at i from node j
			for j in range(len(self.node_array)):
				self.dist_mat[i, j] = self.calc_dist(
					self.node_array[i], self.node_array[j])
				self.power_mat[i, j] = self.calc_power(
					self.node_array[i], self.node_array[j])
				self.connection_mat[i, j] = self.is_connected(
					self.node_array[i], self.node_array[j])
				self.prop_delay_mat[i, j] = self.calc_prop_delay(
					self.node_array[i], self.node_array[j])

		nns.N = self.safedict["N"]
		self.randomize_flying_indices()
		self.Initial_Alpha_Matrix = torch.abs(torch.randn((nns.N, nns.N)))
		for i in range(nns.N):
			for j in range(nns.N):
				if self.connection_mat[i, j] == 0:
					self.Initial_Alpha_Matrix[i, j] = -99e2
			self.node_array[i].Connectivity_Vector = self.connection_mat[:, i]
			self.node_array[i].connections = torch.sum(
				self.node_array[i].Connectivity_Vector)
			self.node_array[i].Initial_Connectivity_Vector = self.connection_mat[:, i]
			self.node_array[i].Propagation_Vector = self.prop_delay_mat[:, i]
			self.node_array[i].prepare_DNN()
		self.Initial_Alpha_Matrix = torch.softmax(
			self.Initial_Alpha_Matrix, dim=0)
		self.Initial_Alpha_Matrix = torch.zeros(
			self.Initial_Alpha_Matrix.shape)
		pass

	def set_mode(self, mode):
		for k in range(nns.N):
			self.node_array[k].DNN.set_both_DNNS_mode(mode)
		self.current_mode = mode
	def is_connected(self, node_A, node_B):
		ID1 = node_A.id
		ID2 = node_B.id
		if self.power_mat[ID1, ID2] > self.Power_Threshold:
			return 1
		else:
			return 0

	def init_deep_parameters(self, force_epochs=100, pars_matrix=None):
		phase_image = []
		period_image = []
		for i in range(nns.N):
			self.node_array[i].DNN.optimizer_Period = torch.optim.SGD(self.node_array[i].DNN.Period_DNN.parameters(),
																	  lr=self.node_array[i].DNN.Period_DNN.lr)
			self.node_array[i].DNN.optimizer_Phase = torch.optim.SGD(self.node_array[i].DNN.Phase_DNN.parameters(),
																	 lr=self.node_array[i].DNN.Phase_DNN.lr)

			if pars_matrix == None:
				pars_vec = torch.tensor(self.safedict["con_graph"][i, :] * (1) / self.safedict["con_graph"][i, :].sum())

			else:
				pars_vec = pars_matrix[i, :]
			for k in range(force_epochs):
				phase_res = self.node_array[i].DNN.Phase_DNN.forward(torch.randn(nns.N * 2))
				loss_phase = ((phase_res - pars_vec) ** 2).sum()
				loss_phase.backward()
				self.node_array[i].DNN.optimizer_Phase.step()
				period_res = self.node_array[i].DNN.Period_DNN.forward(torch.randn(nns.N * 2))
				period_image.append(period_res.detach().numpy())
				loss_period = ((period_res - pars_vec) ** 2).sum()
				loss_period.backward()
				self.node_array[i].DNN.optimizer_Period.step()
			self.node_array[i].DNN.optimizer_Period = torch.optim.Adam(self.node_array[i].DNN.Period_DNN.parameters(),
																	   lr=self.node_array[i].DNN.Period_DNN.lr)
			self.node_array[i].DNN.optimizer_Phase = torch.optim.Adam(self.node_array[i].DNN.Phase_DNN.parameters(),
																	  lr=self.node_array[i].DNN.Phase_DNN.lr)

	def save_data(self):
		json_info = {}
		json_info["Initial_Period"] = self.safedict["Initial_Period"]
		json_info["Initial_Time"] = self.safedict["Initial_Time"]
		json_info["location"] = self.safedict["location"]
		json_info["connectivity"] = self.safedict["con_graph"]
		sp.io.savemat(
			f"pre_generated_setups/{nns.SEED_ID}.mat", json_info)

	def save_times_periods(self, mode):
		times = []
		periods = []
		for k in range(nns.N):
			if type(self.node_array[k].Time) != np.ndarray:
				times.append(self.node_array[k].Time.detach().numpy())
			else:
				times.append(self.node_array[k].Time)
			if type(self.node_array[k].Period) != np.ndarray:
				periods.append(self.node_array[k].Period.detach().numpy())
			else:
				periods.append(self.node_array[k].Period)
		timesperiods = {"t": times, "T": periods}
		sp.io.savemat(
			f"pre_generated_setups/{nns.SEED_ID}_times_periods_{mode}.mat", timesperiods)

	def plot_ONPDRS(self):
		t = []
		T = []
		for k in range(nns.N):
			if type(self.node_array[k].Time) != np.ndarray:
				t.append(self.node_array[k].Time.detach().numpy())
			else:
				t.append(self.node_array[k].Time)
			if type(self.node_array[k].Period) != np.ndarray:
				T.append(self.node_array[k].Period.detach().numpy())
			else:
				T.append(self.node_array[k].Period)
		t = np.array(t)
		T = np.array(T)
		con_mat = self.safedict['con_graph'].astype(bool)
		for node_index in range(con_mat.shape[0]):
			con_mat[node_index, node_index] = 1
			t_temp = (t[con_mat[node_index, :]].T + self.safedict["q_mat"][con_mat[node_index], node_index]).T
			NPDR = (np.max(t_temp, axis=0) - np.min(t_temp, axis=0)) / T[node_index]
			plt.semilogy(NPDR, linewidth=2, label=node_index)
		plt.legend([k for k in range(16)])
		plt.show()

	def calc_power(self, node_A, node_B):
		ID1 = node_A.id
		ID2 = node_B.id
		Power = self.dist_mat[ID1,
		ID2] ** (-self.Power_Exponent) * node_B.TX_Power
		if ID1 == ID2:
			return 0
		else:
			return Power

	def calc_dist(self, node_A, node_B):
		dist = ((node_A.location[0] - node_B.location[0]) ** 2 +
				(node_A.location[1] - node_B.location[1]) ** 2) ** 0.5
		return dist

	def calc_prop_delay(self, node_A, node_B, precalculated=False):
		ID1 = node_A.id
		ID2 = node_B.id
		if precalculated:
			prop_delay = self.safedict["q_mat"][ID1][ID2]
		else:
			prop_delay = self.dist_mat[ID1, ID2] / self.safedict["Speed_Of_Light"]
		return prop_delay

	def prepare_data(self):
		for node_indx in range(self.safedict["N"]):
			self.node_array[node_indx].init_alpha_vec = torch.tensor(
				self.safedict["alpha_mat"])[:, node_indx]

		for batch in range(nns.DPL - 1):
			Network_Times_Vector = torch.zeros(self.safedict["N"], 1)
			for i in range(self.safedict["N"]):
				if batch != 0:
					self.node_array[i].Time[batch] += torch.randn(1)[
														  0] * nns.NOISEPWR
				Network_Times_Vector[i] = self.node_array[i].Time[batch]
			for j in range(self.safedict["N"]):
				self.node_array[j].Time_Step(batch, torch.zeros(
					nns.N), Network_Times_Vector, self.prop_delay_mat[:, j], self.power_mat[:, j], torch.zeros(nns.N),
											 Data_ACQ_Flag=1)
			if batch <= self.safedict["Data_ACQ_Time"] - 2:
				for j in range(self.safedict["N"]):
					self.node_array[j].Data_Diff[:, batch +
													1] = self.node_array[j].Data_Diff[:, batch]

		if nns.PRINT_ENABLED:
			print("Done Preparing Data.")
	def update_locations(self):
		for node_index in self.flying_node_indices:
			self.node_array[node_index].location[0] += np.cos(self.node_array[node_index].flight_direction) * self.node_array[node_index].flight_speed * nns.TNOM
			self.node_array[node_index].location[1] += np.sin(self.node_array[node_index].flight_direction) * self.node_array[node_index].flight_speed * nns.TNOM
		for i in range(len(self.node_array)):
			# row -> reception i here means received power at i from node j
			for j in range(len(self.node_array)):
				self.dist_mat[i, j] = self.calc_dist(
					self.node_array[i], self.node_array[j])
				self.power_mat[i, j] = self.calc_power(
					self.node_array[i], self.node_array[j])
				self.connection_mat[i, j] = self.is_connected(
					self.node_array[i], self.node_array[j])  
				self.prop_delay_mat[i, j] = self.calc_prop_delay(
					self.node_array[i], self.node_array[j])
		for i in range(nns.N):
			self.node_array[i].Connectivity_Vector = self.connection_mat[:, i]
			self.node_array[i].connections = torch.sum(
				self.node_array[i].Connectivity_Vector)
			self.node_array[i].Propagation_Vector = self.prop_delay_mat[:, i]
		for i in range(len(self.node_array)):
			self.node_array[i].CData[:, 0] = self.power_mat[:, i]


	def randomize_flying_indices(self):
		self.flying_node_indices = []
		directions_radians = [0.7507827351314131, 5.842510342614021, 2.095378319093055, 2.8916073787320844]
		node_indices 	   = [10, 5, 12, 14]
		c = 0
		self.flying_node_indices = node_indices
		for k in node_indices:
			self.node_array[k].flight_direction = directions_radians[c]
			self.node_array[k].flight_speed = nns.flying_node_speed
			c += 1

		print([0.7507827351314131, 5.842510342614021, 2.095378319093055, 2.8916073787320844])
		print([10, 5, 12, 14])

		print([self.node_array[k].flight_direction for k in self.flying_node_indices])
		print([k for k in self.flying_node_indices])



		print([360 / (2 * np.pi) * self.node_array[k].flight_direction for k in self.flying_node_indices])

		print("")

	def run_network(self, batches, Deep_Enabled=True):

		Network_Times_Vector = torch.zeros(self.safedict["N"], 1)
		Network_Times_Diff_Vec = torch.zeros(
			self.safedict["N"], self.safedict["N"])
		nns.N = self.safedict["N"]
		Alpha_Vectors = torch.zeros(self.safedict["N"] * 2)
		Phase_Alpha = Alpha_Vectors[nns.N:]
		Period_Alpha = Alpha_Vectors[:nns.N]
		nns.RANDOMIZE_LOCATIONS = True
		for batch in range(batches):

			if (batch > nns.start_flying_index) and (batch < nns.stop_flying_index):
				self.update_locations() 
			if batch != 0:
				for noise_index in range(nns.N):
					self.node_array[noise_index].Time[batch] += torch.randn(1)[
																	0] * nns.NOISEPWR
			Transmitter = np.mod(batch, self.safedict["N"])
			Network_Times_Vector[Transmitter] = self.node_array[Transmitter].Time[batch]

			for j in range(self.safedict["N"]):
				if Deep_Enabled and (Transmitter == self.safedict["N"] - 1):
					Network_Times_Diff_Vec[j, Transmitter] = Network_Times_Vector[Transmitter] - \
															 self.node_array[j].Time[batch]
					Phase_Alpha = self.node_array[j].DNN.Phase_DNN(torch.cat(
						(Network_Times_Diff_Vec[j, :] + self.prop_delay_mat[:,j], self.node_array[j].CData[:, 0]), dim=0)) # Changed here!
					Period_Alpha = self.node_array[j].DNN.Period_DNN(torch.cat(
						(self.node_array[j].Period_Ob - self.node_array[j].Period[batch],
						 self.node_array[j].CData[:, 0]), dim=0))
				elif not (Deep_Enabled) and (Transmitter == self.safedict["N"] - 1):
					Phase_Alpha = torch.tensor(
						self.node_array[j].phase_alpha_vector)
					Period_Alpha = torch.tensor(
						self.node_array[j].period_alpha_vector)

				self.node_array[j].Time_Step(batch, Phase_Alpha, Network_Times_Vector,
											 self.prop_delay_mat[:,
											 j], self.power_mat[:, j], Period_Alpha,
											 Data_ACQ_Flag=0)

	# print("Network Ran.")
	def continue_nodes(self):
		for node_ in self.node_array:
			node_.Period[0] = node_.Period[self.safedict["Data_ACQ_Time"]]
			node_.Time[0] = torch.remainder(node_.Time[self.safedict["Data_ACQ_Time"]], node_.Period[0])

	def start_training(self):
		with concurrent.futures.ProcessPoolExecutor() as executor:
			self.futures = [executor.submit(self.node_array[i].train_node) for i in range(self.safedict["N"])]
			concurrent.futures.wait(self.futures)  # Wait for all the nodes to finish training
		self.node_array = [pickle.loads(f.result()) for f in self.futures]

	def start_training_recursive_root(self, training_node_indices, first_iter_flag):
		with concurrent.futures.ProcessPoolExecutor(max_workers=nns.amount_of_cores) as executor:
			self.futures = [executor.submit(self.node_array[i].train_node, first_iter_flag) for i in
							training_node_indices]
			concurrent.futures.wait(self.futures)  # Wait for all the nodes to finish training
		self.node_array_temp_unstructured = [pickle.loads(f.result()) for f in self.futures]
		for i in range(len(self.node_array_temp_unstructured)):
			ID = self.node_array_temp_unstructured[i].ID
			self.node_array[ID] = self.node_array_temp_unstructured[i]
		training_node_indices = []
		for i in range(nns.N):
			if self.node_array[i].Learning_Threshold_Failed:
				training_node_indices.append(i)
		return training_node_indices

	def start_training_recursive(self):
		first_iter_flag = True
		training_node_indices = [i for i in range(nns.N)] 
		counts = 0
		ONPDRS = np.array([self.node_array[k].calc_NPDR() for k in range(nns.N)])
		First_ONPDRS = np.array([self.node_array[k].calc_NPDR() for k in range(nns.N)]) + 0
		while len(training_node_indices) != 0 and nns.DLI > counts:
			self.failed_node_indices = []
			training_node_indices = self.start_training_recursive_root(training_node_indices,
																	   first_iter_flag)  # ONLY THOSE WHO FAILED
			first_iter_flag = False
			counts += 1
			self.prepare_data()  # actually, it's evaluation.f
			

	def plot_periods(self, end, lw=1):
		for i in range(self.safedict["N"]):
			plt.plot(self.node_array[i].Period.detach().numpy()[1:end], lw=lw)
		pass

	def plot_periods_skew(self, end):
		periods = np.zeros((nns.N, end))
		for i in range(self.safedict["N"]):
			periods[i, :] = self.node_array[i].Period.detach().numpy()[0:end]
		pass
		plt.plot((periods - np.mean(periods, axis=0)).T)

	def plot_periods_average(self, end):
		periods = np.zeros((nns.N, end))
		for i in range(self.safedict["N"]):
			periods[i, :] = self.node_array[i].Period.detach().numpy()[0:end]
		pass
		plt.plot((np.mean(periods, axis=0)))

	def plot_phases(self, end):
		nns.N = self.safedict["N"]
		Times = np.zeros((nns.N, end))
		for i in range(self.safedict["N"]):
			Times[i, :] = self.node_array[i].Time[:end].detach().numpy()
		plt.plot((Times - np.mean(Times, 0)).T)

	def plot_connections(self, ID_2color=17):
		con_mat = self.connection_mat.detach().numpy()
		G = nx.DiGraph()
		color_vec = []
		style_vec = []
		for ID in range(self.safedict["N"]):
			for ID2 in range(self.safedict["N"]):
				if con_mat[ID, ID2] == 1:
					if ID == ID_2color:
						G.add_edge(ID, ID2)
						color_vec.append(3)
						style_vec.append("-")
					else:
						color_vec.append(3)
						G.add_edge(ID, ID2)
						style_vec.append("-")
		labels = {}
		for ID in range(self.safedict["N"]):
			labels[ID] = f"{ID}"
		ax = plt.axes()
		ax.set_axis_on()
		ax.grid('on')
		nx.draw(G, np.array(self.safedict["location"]), labels=labels, font_size=16, node_size=800,
				ax=ax, width=2.0, edge_color=color_vec)  # Set width to 2.0
		ax.set_axis_on()
		ax.tick_params(left=True, bottom=True,
					   labelleft=True, labelbottom=True)
		plt.xlabel("X coordinate [m]")
		plt.ylabel("Y coordinate [m]")
		plt.show()

	def classic_alpha(self):
		for ID in range(self.safedict["N"]):
			denom = 0

			self.node_array[ID].phase_alpha_vector = torch.zeros(
				self.safedict["N"])
			for ID2 in range(self.safedict["N"]):
				if self.is_connected(self.node_array[ID], self.node_array[ID2]):
					denom += self.calc_power(
						self.node_array[ID], self.node_array[ID2])
			for ID2 in range(self.safedict["N"]):
				if self.is_connected(self.node_array[ID], self.node_array[ID2]):
					self.node_array[ID].phase_alpha_vector[ID2] = self.calc_power(
						self.node_array[ID], self.node_array[ID2]) / denom
			self.node_array[ID].period_alpha_vector = self.node_array[ID].phase_alpha_vector

	def random_alpha(self):
		phase_alpha_mat, _ = gen_random_softmax_weights(
			self.safedict["con_graph"])
		period_alpha_mat, _ = gen_random_softmax_weights(
			self.safedict["con_graph"])
		for ID in range(self.safedict["N"]):
			self.node_array[ID].phase_alpha_vector = phase_alpha_mat[:, ID]  #
		for ID in range(self.safedict["N"]):
			self.node_array[ID].period_alpha_vector = period_alpha_mat[:, ID]  #

	def Aggregate_Times(self, sample_start=1, sample_end=100):
		Time_Mat = torch.zeros(self.safedict["N"], sample_end - sample_start)
		for node_indx in range(self.safedict["N"]):
			Time_Mat[node_indx,
			:] = self.node_array[node_indx].Time[sample_start: sample_end]
		return Time_Mat

	def generated_observed_periods(self):
		for i in range(self.safedict["N"]):
			k = 0
			observed_sum = 0
			for j in range(self.safedict["N"]):
				if self.node_array[i].Connectivity_Vector[j] == 1:
					k += 1
					observed_sum += self.node_array[j].Period[0]
			if k > 0:
				self.node_array[i].observed_mean_period = observed_sum / k
			else:
				self.node_array[i].observed_mean_period = 0

	def save_sim_data(self):

		sp.io.savemat(
			f"/content/drive/MyDrive/sim_data/Generated_Solution_{nns.SEED_ID}.mat", self.safedict)


def create_node_array(sim_dict):
	Nodes_Array = []
	for id in range(sim_dict["N"]):
		Nodes_Array.append(node(id, sim_dict))

	return Nodes_Array

def load_data(SEED_ID, safedict):
    json_info = sp.io.loadmat(
        f"pre_generated_setups/{SEED_ID}.mat")
    for key in json_info.keys():
        safedict[key] = json_info[key]
    return safedict

if __name__ == '__main__':
	torch.set_default_tensor_type(torch.DoubleTensor)
	logging.basicConfig(level=logging.INFO,
						format='%(asctime)s: %(levelname)s: %(message)s',
						stream=sys.stdout)
	SEED_ID = nns.SEED_ID
	np.random.seed(SEED_ID)


	plt.rcParams['savefig.bbox'] = 'tight'
	plt.rcParams['savefig.format'] = 'svg'
	plt.rcParams['figure.frameon'] = False
	plt.rcParams['figure.figsize'] = (16, 12)
	plt.rcParams['figure.dpi'] = 150
	plt.rcParams['figure.facecolor'] = [232 / 240, 232 / 240, 232 / 240]
	plt.rcParams['figure.edgecolor'] = 'white'

	plt.rcParams['axes.prop_cycle'] = plt.cycler(
		color=['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE'])  # Matlab styled!
	plt.rcParams['axes.facecolor'] = 'white'
	plt.rcParams['axes.edgecolor'] = 'black'
	plt.rcParams['axes.linewidth'] = 1
	plt.rcParams['axes.grid'] = False
	plt.rcParams['axes.grid.which'] = 'both'
	plt.rcParams['axes.titlesize'] = 'large'
	plt.rcParams['axes.titleweight'] = 'normal'
	plt.rcParams['axes.axisbelow'] = 'line'

	plt.rcParams['lines.linewidth'] = 0.5

	plt.rcParams['font.family'] = 'Times'
	plt.rcParams['font.size'] = 14
	plt.rcParams['font.weight'] = 'normal'

	plt.rcParams['xtick.top'] = True
	plt.rcParams['xtick.major.size'] = 16
	plt.rcParams['xtick.major.width'] = 0.6
	plt.rcParams['xtick.direction'] = 'in'

	plt.rcParams['ytick.major.size'] = 16
	plt.rcParams['ytick.major.width'] = 0.6
	plt.rcParams['ytick.right'] = True
	plt.rcParams['ytick.direction'] = 'in'

	plt.rcParams['legend.loc'] = 'upper right'
	plt.rcParams['legend.shadow'] = False
	plt.rcParams['legend.fontsize'] = 12

	F_Nom = 1 / nns.TNOM
	fa = F_Nom * (1 - 10 ** (-6) * nns.PPM_VAR)
	fb = F_Nom * (1 + 10 ** (-6) * nns.PPM_VAR)
	l0 = 0.29
	l1 = 0.31
	safedict = Temporary_Dictionary_Preparation()
	TXP = safedict["TX_Power"]
	Link_Passed = False
	node_index = 0
	safedict["SEED"] = SEED_ID
	## Testing purposes - Shuffle seed!

	if nns.RANDOMIZE_INITS:

		safedict["Initial_Period"] = 1. / F_lambda(fa, fb, nns.N)
		safedict["Initial_Time"] = nns.Initial_Time_Scaling * np.random.rand(
			nns.N, 1) * safedict["Initial_Period"]
	## Perturbation TEST!
	# safedict["Initial_Time"][13] = safedict["Initial_Time"][13]*0.3
	## Perturbation test.
	else:
		safedict["Initial_Period"] = np.array(safedict["Initial_Period"])
		safedict["Initial_Time"] = np.array(safedict["Initial_Time"])

	while (Link_Passed == False):

		safedict["TX_Power"] = TXP

		if nns.RANDOMIZE_LOCATIONS:
			safedict["location"] = 10e3 * np.random.rand(nns.N, 2)
		else:
			safedict["location"] = np.array(safedict["location"])

		q_mat, simeone_alphas, con_graph = get_q_alphas_connectivity(
			safedict["location"], safedict["Power_Exponent"], safedict["Power_Threshold"], safedict["TX_Power"][0],
			safedict["Speed_Of_Light"])
		# np.random.seed(None)
		alpha_mat_random, fc_flag = gen_random_softmax_weights(con_graph)
		if fc_flag:
			link_amount = np.sum(con_graph)
			Link_Passed = True
			if link_amount < nns.links_l0 * nns.N * (nns.N - 1):
				TXP = [2 for k in TXP]
				# print(
				# f"this network has less than {l0 * 100}% links, retrying!!, TX Power is now {TXP}")
				Link_Passed = False
			if link_amount > nns.links_l1 * nns.N * (nns.N - 1):
				TXP = [2 for k in TXP]
				# print(
				# f"this network has more than {l1 * 100}% links, retrying!!, TX Power is now {TXP}")
				Link_Passed = False


	safedict["alpha_mat"] = alpha_mat_random
	safedict["con_graph"] = con_graph
	safedict["q_mat"] = q_mat
	safedict["simeone_alphas"] = simeone_alphas
	if nns.RANDOMIZE_INITS:
		# np.random.seed(None)

		safedict["Initial_Period"] = 1. / F_lambda(fa, fb, nns.N)
		safedict["Initial_Time"] = nns.Initial_Time_Scaling * np.random.rand(
			nns.N, 1) * safedict["Initial_Period"]

	else:
		safedict["Initial_Period"] = np.array(safedict["Initial_Period"])
		safedict["Initial_Time"] = np.array(safedict["Initial_Time"])

	if nns.load_loc_from_file:
		safedict["location"] = np.loadtxt(f"Locations_{nns.SEED_ID}.npy")

	q_mat, simeone_alphas, con_graph = get_q_alphas_connectivity(
		safedict["location"], safedict["Power_Exponent"], safedict["Power_Threshold"], safedict["TX_Power"][0],
		safedict["Speed_Of_Light"])
	safedict["alpha_mat"] = alpha_mat_random
	safedict["con_graph"] = con_graph
	safedict["q_mat"] = q_mat
	safedict["simeone_alphas"] = simeone_alphas

	Node_Aggregation = node_aggregation(safedict)
	Node_Aggregation.Assemble_Properties()
	Node_Aggregation.set_mode("min_max_avg_neutralize")

	Node_Aggregation.prepare_data()

	Node_Aggregation.save_data()
	Node_Aggregation.continue_nodes()
	Node_Aggregation.prepare_data()

	SIM_LEN = safedict["sim_length"] - 2
	logcell = {
		"NPD_Range": {"Deep": 0.0, "Random": 0.0},
		"Mean_Period": {"Deep": 0.0, "Random": 0.0},
		"STD_Period": {"Deep": 0.0, "Random": 0.0},
		"STD_NPD": {"Deep": 0.0, "Random": 0.0},
		"Mean_Phase": {"Deep": 0.0, "Random": 0.0},
		"STD_Phase": {"Deep": 0.0, "Random": 0.0},
		"SPM": list(),
		"CM": list(),
		"Failed_Nodes": list()}

	# print("Network starting to train.")
	Node_Aggregation.start_training_recursive()
	# print("Network stopped training.")

	SIM_LEN = safedict["sim_length"] - 2
	print("Finished training, running!")
	Node_Aggregation.run_network(SIM_LEN, Deep_Enabled=True)
	Node_Aggregation.save_times_periods("Deep_Learning_Plus_MinMax")
