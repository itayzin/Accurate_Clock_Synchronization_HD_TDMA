import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch import nn
import logging
import node_network_settings as nns
import pickle
torch.set_default_tensor_type(torch.DoubleTensor)
from copy import deepcopy

class HadamardLayer(nn.Module):
    def __init__(self, N):
        super(HadamardLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(N))  # initialize N weights

    def forward(self, x):
        return x * self.weights  # element-wise (Hadamard) multiplication

class node():
    def __init__(self, ID, sim_properties_dict):
        self.saved_times = []
        self.saved_periods = []
        self.period_alpha_vector = torch.rand(nns.N) / 121512211221
        self.phase_alpha_vector  = torch.rand(nns.N) / 121222122112
        self.safedict = sim_properties_dict
        self.ID = ID
        self.id = ID
        self.location = sim_properties_dict["location"][ID]
        self.TX_Power = sim_properties_dict["TX_Power"][ID]
        self.eps = sim_properties_dict["eps"][ID]
        self.first_loss_phase = -1
        self.first_loss_period = -1
        self.first_loss_ratio = 1
        self.first_epochs_loss = "Not Calculated Yet"
        self.period_eps = sim_properties_dict["period_eps"][ID]
        self.Period = torch.zeros(
            sim_properties_dict["sim_length"], requires_grad=False)
        self.Period[0] = torch.tensor(
            sim_properties_dict["Initial_Period"][ID][0])
        self.Period_Fake = sim_properties_dict["Data_ACQ_Time"] * [
            torch.zeros(1)]
        self.Period_Fake[0] = torch.tensor(
            [self.Period[0].item()], requires_grad=False)
        self.Time = torch.zeros(
            sim_properties_dict["sim_length"], requires_grad=False)
        self.Time[0] = torch.tensor(sim_properties_dict["Initial_Time"][ID][0])
        # To avoid inplace errors - Making list of tensors rather than tensor.
        self.Time_Fake = sim_properties_dict["Data_ACQ_Time"] * \
            [torch.zeros(1, requires_grad=False)]
        self.Time_Fake[0] = torch.tensor([self.Time[0]])
        self.Data = torch.zeros(
            sim_properties_dict["N"], sim_properties_dict["Data_ACQ_Time"], requires_grad=False)
        self.Data_Period = torch.zeros(
            sim_properties_dict["N"], sim_properties_dict["Data_ACQ_Time"], requires_grad=False)
        self.CData = torch.zeros(
            sim_properties_dict["N"], sim_properties_dict["Data_ACQ_Time"], requires_grad=False)
        self.Data_Diff = self.Data.clone()
        self.Network_Times_Vector = torch.zeros(
            sim_properties_dict["N"], requires_grad=False)
        self.Network_Times_Vector_Fake = torch.zeros(
            sim_properties_dict["N"], requires_grad=False)
        self.Data_Transmitter = np.zeros(sim_properties_dict["Data_ACQ_Time"])
        self.Period_Ratio = 0.05
        self.Fixed_Period_Bool = False
        self.Fixed_Period = self.Period[0]
        self.Time_Diff = sim_properties_dict["N"] * [torch.zeros(1)]
        self.Time_Diff_Fake = sim_properties_dict["N"] * [torch.zeros(1)]
        self.Period_Ob = torch.zeros(nns.N, requires_grad=False)
        self.Period_Ob_Fake = torch.zeros(nns.N, requires_grad=False)
        self.Period_Update = torch.tensor(0, requires_grad=False)
        self.Period_Fake_Update = torch.tensor(0, requires_grad=False)
        self.bias_correction = 0
        self.loss_phase_vec = []
    def prepare_DNN(self):
        self.DNN = CreateSimpleDnn(self.safedict["N"] * 2, self, lr=0.01)

    def self_inits(self, Period, Phase, location, ID):
        pass

    def clean_time_fake(self):
        self.Period_Fake = self.safedict["Data_ACQ_Time"] * \
            [torch.zeros(1, requires_grad=False)]
        self.Period_Fake[0] = torch.tensor([self.Period[0].item()])
        self.Time_Fake = self.safedict["Data_ACQ_Time"] * \
            [torch.zeros(1, requires_grad=False)]
        self.Time_Fake[0] = torch.tensor([self.Time[0]])
        self.Time_Diff = self.safedict["N"] * \
            [torch.zeros(1, requires_grad=False)]
        self.Time_Diff_Fake = self.safedict["N"] * \
            [torch.zeros(1, requires_grad=False)]
        self.Period_Ob_Fake = self.safedict["N"] * \
            [torch.zeros(1, requires_grad=False)]
    def train_node(self, first_iter_flag):

        if not first_iter_flag:
            self.period_alpha_vector = torch.rand(nns.N) / 3
            self.phase_alpha_vector = torch.rand(nns.N) / 3
            self.DNN = CreateSimpleDnn(self.safedict["N"] * 2, self, lr=0.01)
        self.DNN.batches = self.safedict["Data_ACQ_Time"] - 2 * nns.N
        self.DNN.batches = self.DNN.batches - 1
        self.DNN.loss_capture_vector = []
        self.DNN.loss_capture_vector_phase = []
        self.DNN.loss_capture_vector_period = []
        self.DNN.epochs = nns.EPOCHS
        self.DNN.weight_history_phase = []
        self.DNN.weight_history_period = []
        self.DNN.optimizer_Phase = torch.optim.Adam(
            self.DNN.Phase_DNN.parameters(), lr=self.DNN.Phase_DNN.lr)
        self.DNN.optimizer_Period = torch.optim.Adam(
            self.DNN.Period_DNN.parameters(), lr=self.DNN.Period_DNN.lr)
        logging.info(f"Node #{self.ID}'s DNN started training.")
        self.DNN.Phase_DNN.train()
        self.DNN.Period_DNN.train()
        for Epoch in range(self.DNN.epochs):
            self.DNN.Epoch = Epoch
            self.DNN.Train(
                self.Data, self.CData, self.Data_Diff)
            if self.first_epochs_loss == "Not Calculated Yet":
                if Epoch == 10:
                    self.first_epochs_loss = self.get_avg_loss('start')
        if self.first_epochs_loss * nns.LOSS_TH < self.get_avg_loss('end'):
            self.Learning_Threshold_Failed = True
            print(f"Node {self.ID} Failed training to at-least {nns.LOSS_TH * 100}% of its initial loss.\n it had a loss of {self.get_avg_loss('end')} compared to {self.first_epochs_loss}")
        else :
            print(f"Node {self.ID} Succeeded in training to at-least {nns.LOSS_TH * 100}% of its initial loss.\n it had a loss of {self.get_avg_loss('end')} compared to {self.first_epochs_loss}")
            self.Learning_Threshold_Failed = False

        self.DNN.Phase_DNN.eval()
        self.DNN.Period_DNN.eval()
        logging.info(f"Node #{self.ID}'s DNN stopped training. Plotting loss function(s)!")
        logging.info(f"Node #{self.ID}'s loss :")
        serialized_node = pickle.dumps(self)  # serialize the node object
        return serialized_node
    def get_avg_loss(self, when):
        if when == "start":
            result = np.array(self.DNN.loss_capture_vector[0]).mean()
        elif when == "end":
            result = np.array(self.DNN.loss_capture_vector[-1]).mean()
        else :
            raise Exception(f"State not defined when = {when}")

        return result

    def update_connectivity_DNN(self, Connectivity_Vector):
        Connectivity_Vector *= self.Initial_Connectivity_Vector
        self.DNN.Period_DNN.Wrapping_Layers_Input.weight = torch.nn.Parameter(torch.diag(
            torch.cat((Connectivity_Vector, Connectivity_Vector), dim=0)))
        self.DNN.Period_DNN.Wrapping_Layers_Input.weight.requires_grad = False
        self.DNN.Period_DNN.Wrapping_Layers_Output.weight = torch.nn.Parameter(
            torch.diag(Connectivity_Vector))
        self.DNN.Period_DNN.Wrapping_Layers_Output.weight.requires_grad = False

        self.DNN.Phase_DNN.Wrapping_Layers_Input.weight = torch.nn.Parameter(torch.diag(
            torch.cat((Connectivity_Vector, Connectivity_Vector), dim=0)))
        self.DNN.Phase_DNN.Wrapping_Layers_Input.weight.requires_grad = False
        self.DNN.Phase_DNN.Wrapping_Layers_Output.weight = torch.nn.Parameter(
            torch.diag(Connectivity_Vector))
        self.DNN.Phase_DNN.Wrapping_Layers_Output.weight.requires_grad = False
    def Time_Step(self, k, Alpha_Vector, Network_Times_Vector, Propagation_Vector, Network_Powers, Period_Alpha_Vector,
                  Data_ACQ_Flag=0):
        Transmitter = np.mod(k, self.safedict["N"])
        self.update_connectivity_DNN(self.Connectivity_Vector)
        if np.mod((k) // (nns.N), 3) == 0:
            mode_ = "period"
            peps_0 = nns.PERIOD_EPS_0

            eps_0 = nns.NULL_PHASE_EPS_0  # nns.EPS_0
        elif np.mod((k) // (nns.N), 3) == 1:
            mode_ = "phase"
            peps_0 = nns.NULL_PERIOD_EPS_0  # nns.PERIOD_EPS_0
            eps_0 = nns.EPS_0
        else:
            mode_ = "cooldown"
            peps_0 = nns.NULL_PERIOD_EPS_0
            eps_0 = nns.NULL_PHASE_EPS_0  # nns.EPS_0
        if np.mod(k, self.safedict["N"] - 1 + 1) == self.safedict["N"] - 1:
            Updating = True
            with torch.no_grad():
                self.Period_Ob[Transmitter] = (
                    Network_Times_Vector[Transmitter] - self.Network_Times_Vector[
                        Transmitter]) / nns.N
                self.Network_Times_Vector[np.mod(
                    k, self.safedict["N"])] = Network_Times_Vector[np.mod(k, self.safedict["N"])]

                self.Time_Diff[np.mod(k, self.safedict["N"])] = self.Network_Times_Vector[np.mod(
                    k, self.safedict["N"])] - self.Time[k]
                Phase_Alpha = self.DNN.Phase_DNN(torch.cat((torch.tensor(
                    self.Time_Diff) +Propagation_Vector, self.CData[:, 0]), dim=0))
                if mode_ == "phase" and k > (nns.Data_ACQ_Time - 100):
                    self.phase_connectivity_history = Phase_Alpha
                Period_Alpha = self.DNN.Period_DNN(torch.cat(
                    (self.Period_Ob.clone().detach() - self.Period[k], self.CData[:, 0]), dim=0))
                if mode_ == "period" and k > (nns.Data_ACQ_Time - 100):
                    self.period_connectivity_history = Period_Alpha

                if k > nns.N:
                    self.Period_Update = (peps_0 * torch.dot(
                        Period_Alpha, (self.Period_Ob.clone().detach() - self.Period[k-nns.N])))

                    self.Period[k + 1] = self.Period[k] +  self.Period_Update / nns.N
                else :
                    self.Period[k + 1] = self.Period[k]
                self.Time[k + 1] = self.Time[k] + self.Period[k] + eps_0 * (torch.dot(Phase_Alpha, (torch.Tensor(
                    self.Time_Diff) + Propagation_Vector))) 
        else:
            with torch.no_grad():
                Updating = False
                if k > 2 * nns.N:
                    self.Period[k + 1] = self.Period[k] +  self.Period_Update / nns.N #
                else:
                    self.Period[k + 1] = self.Period[k]
                self.Period_Ob[Transmitter] = (
                    Network_Times_Vector[Transmitter] - self.Network_Times_Vector[
                        Transmitter]) / nns.N
                self.Network_Times_Vector[np.mod(
                    k, self.safedict["N"])] = Network_Times_Vector[np.mod(k, self.safedict["N"])]
                self.Time_Diff[np.mod(k, self.safedict["N"])] = self.Network_Times_Vector[np.mod(
                    k, self.safedict["N"])] - self.Time[k]
                self.Time[k + 1] = self.Time[k] + self.Period[k]
            with torch.no_grad():
                self.Period[k + 1] = self.Period[k + 1] + self.bias_correction

        if Data_ACQ_Flag and k <= self.safedict["Data_ACQ_Time"] - 2:
            self.Data[:, k] = self.Network_Times_Vector + Propagation_Vector
            self.Data_Transmitter[k] = Transmitter
            self.CData[:, k] = Network_Powers
            self.Data_Diff[:, k] = self.Data_Diff[:, k - 1]
            self.Data_Diff[Transmitter, k] = self.Network_Times_Vector[np.mod(
                k, self.safedict["N"])] + Propagation_Vector[np.mod(k, self.safedict["N"])] - self.Time[
                k]  
            self.Data_Period[:, k] = self.Period_Ob
        pass

    def calc_loss_phase(self):
        res = 0
        for i in range(nns.N):
            if self.Connectivity_Vector[i] == 1:
                res += self.Time_Diff[i] ** 2 / self.Connectivity_Vector.sum()

        self.loss_phase_vec.append(res)
   



class CreateSimpleDnn():
    def __init__(self, Inputs, Node: node, lr=10, epochs=10, batches=-1):
        self.min_loss_sum_period = 1e9
        self.min_loss_sum_phase = 1e9
        self.loss_ratio = 1
        self.Node = Node
        # forced_init_weights    = torch.cat((,), dim=0)
        # torch.manual_seed(9596545 + self.Node.ID)
        self.Epochs_Per_Half  = 5
        self.Phase_DNN = Simple_Phase_DNN(
            Inputs, Node.Connectivity_Vector, self.Node.period_alpha_vector.clone().detach(), self.Node.safedict, self.Node.ID)
        self.Period_DNN = Simple_Period_DNN(
            Inputs, Node.Connectivity_Vector, self.Node.phase_alpha_vector.clone().detach(), self.Node.safedict, self.Node.ID)
        self.Phase_DNN.lr = nns.PHASE_LR
        self.Period_DNN.lr = nns.PERIOD_LR
        self.optimizer_Phase = torch.optim.Adam(
            self.Phase_DNN.parameters(), lr=self.Phase_DNN.lr)
        self.optimizer_Period = torch.optim.Adam(
            self.Period_DNN.parameters(), lr=self.Period_DNN.lr)
        self.epochs = nns.EPOCHS
        self.weight_history_phase = []
        self.weight_history_period = []
        self.Phase_DNN.eval() # For Data preparation
        self.Period_DNN.eval() # For Data preparation
        if batches == -1:
            self.batches = Node.safedict["Data_ACQ_Time"] - 2 * nns.N
        self.batches = self.batches - 1
        self.loss_capture_vector = []
        self.loss_capture_vector_phase = []
        self.loss_capture_vector_period = []

    def set_both_DNNS_mode(self, mode):
        self.Period_DNN.layer_mode = mode
        self.Phase_DNN.layer_mode  = mode
    def FModel(self, input):
        Phase_Result = self.Phase_DNN(input)
        Period_Result = self.Period_DNN(input)
        return torch.cat((Period_Result, Phase_Result), 0)

    def Freeze_DNN(self, DNN):
 

        for param in DNN.parameters():
            if id(param) not in [id(perm_parm) for perm_parm in DNN.Permafrost_parameters]:
                if param.requires_grad:
                    param.requires_grad = False
                    param.grad = None

    def Unfreeze_DNN(self, DNN):


        for param in DNN.parameters():
            if id(param) not in [id(perm_parm) for perm_parm in DNN.Permafrost_parameters]:
                if ~param.requires_grad:
                    param.requires_grad = True

    def plot_loss_func(self):
        plt.subplot(211)
        indices_a = [a for a in range(
            len(self.loss_capture_vector[:nns.EPOCHS // 2 - 1]))]
        plt.plot(indices_a, np.array(
            self.loss_capture_vector[:nns.EPOCHS // 2 - 1]))
        plt.title("Loss Func (Period)")
        plt.subplot(212)
        indices_b = [
            a + nns.EPOCHS // 2 for a in range(len(self.loss_capture_vector[:nns.EPOCHS // 2 - 1]))]
        plt.plot(indices_b, np.array(
            self.loss_capture_vector[nns.EPOCHS // 2 + 1:]))
        plt.title("Loss Func (Phase)")

    def loss_func_phase(self, Data):
        loss_sum = 0
        self.loss_int_vec_phase= []
        for batch in range(nns.N, self.batches):
            transmitter = np.mod(batch, self.Node.safedict["N"])
            if np.mod((batch) // (nns.N), 3) == 0:
                loss_enabled = 0
            elif np.mod((batch) // (nns.N), 3) == 1:
                loss_enabled = 1
            else:
                loss_enabled = 0
            if self.Node.Connectivity_Vector[transmitter] == 1:

                instantenous_loss = (
                    (self.Node.Time[batch] - Data[transmitter, batch])**2)
                loss_sum +=  instantenous_loss #* sign  # Fixed for TDMA
            if nns.EPOCHPLOT:
                if np.mod(batch, nns.N) == nns.N - 1:
                    if len(self.loss_int_vec_phase) ==0:
                        self.loss_int_vec_phase.append(
                            loss_sum.clone().detach().numpy())
                    else :
                        self.loss_int_vec_phase.append(
                            loss_sum.clone().detach().numpy() - self.loss_int_vec_phase[-1])
        if self.Node.first_loss_phase == - 1:
            self.Node.first_loss_phase = loss_sum.clone().detach()
        if nns.EPOCHPLOT:
            pass
          
        if loss_sum < self.min_loss_sum_phase:
            self.min_Phase_DNN = deepcopy(self.Phase_DNN)
            self.min_loss_sum_phase = loss_sum.clone()
            self.copied_min_loss_sum_phase = self.min_loss_sum_phase + 1e-15
        else:
            pass
        return loss_sum

    def loss_func_period(self, Data):
        loss_sum = 0
        self.loss_int_vec_period = []
        for batch in range(2 * nns.N, self.batches):
            if np.mod((batch) // (nns.N), 3) == 0:
                loss_enabled = 1
            elif np.mod((batch) // (nns.N), 3) == 1:
                loss_enabled = 0
            else:
                loss_enabled = 0
            transmitter = np.mod(batch, self.Node.safedict["N"])
            if self.Node.Connectivity_Vector[transmitter] == 1:
                instantenous_loss =  ((self.Node.Period[batch - nns.N] - (Data[transmitter, batch] - Data[transmitter, batch - nns.N]) / nns.N)**2)  # Calculate Period of observed node, (it is over 16 timeslots )


                loss_sum +=  instantenous_loss  #* sign# Fixed for TDMA
            if nns.EPOCHPLOT:
                if np.mod(batch, nns.N) == nns.N - 1:
                    if len(self.loss_int_vec_period) ==0:
                        self.loss_int_vec_period.append(
                            loss_sum.clone().detach().numpy())
                    else :
                        self.loss_int_vec_period.append(
                            loss_sum.clone().detach().numpy() - self.loss_int_vec_period[-1])

        if loss_sum < self.min_loss_sum_period:
            self.min_Period_DNN = deepcopy(self.Period_DNN)
            self.min_loss_sum_period = loss_sum.clone()
            self.copied_min_loss_sum_period = self.min_loss_sum_period + 1e-15
        else:
            pass
        return loss_sum
class Weighted_Averaging_Learner(nn.Module):
    def __init__(self):
        super().__init__()
        eps = 0.1
        self.weight = nn.Parameter(torch.tensor([1 - eps, eps]), requires_grad=True)

    def forward(self, x):
        interm_weights = self.weight
        return interm_weights[0] * x[:nns.N] + interm_weights[1] * x[nns.N:]

class Learnable_Epsilon(nn.Module):
    def __init__(self, init_epsilon):
        super().__init__()
        init_value = torch.log(torch.tensor(init_epsilon / (1 - init_epsilon)))
        self.weight = nn.Parameter(torch.tensor(init_value), requires_grad=True)
    def forward(self, x):
        interm_weights = 1 / nns.EPS_0 * torch.sigmoid(self.weight)
        return interm_weights * x


class NormalizationLayer(nn.Module):
    def forward(self, x):
        if x.sum() == 0:
            return torch.zeros(x.shape)
        x = nn.functional.relu(x)
        sum_inputs = x.sum()
        normalized = x / sum_inputs
        return normalized

class NormalizedSoftmax(nn.Module):
    def forward(self, x, neutralize):
        if neutralize == "no_neutralize":
            x = x[:nns.N]
            x = x / x.abs().sum()
            x[x == 0] = -99999.9
            argmax_ = torch.argmax(x, dim=0)
            x = nn.functional.one_hot(argmax_, nns.N)
            return x
        else :
            x = x[:nns.N]
            x = x / x.abs().sum()
            x[x < 0] = 0
            if len(x[x>0]) == 0:
                return torch.zeros(x.shape)
            x[x == 0] = -99999.9
            argmax_ = torch.argmax(x, dim=0)
            x = nn.functional.one_hot(argmax_, nns.N)
            return x
class NormalizedSoftmin(nn.Module):
    def forward(self, x, neutralize):
        if neutralize == "no_neutralize":
            x = x[:nns.N]
            x = x / x.abs().sum()

            x[x == 0] = 99999.9
            argmin_ = torch.argmin(x, dim=0)
            x = nn.functional.one_hot(argmin_, nns.N)
            return x
        else:
            x = x[:nns.N]
            x = x / x.abs().sum()
            x[x>0] = 0 # So we won't go down!
            if len(x[x<0]) == 0:
                return torch.zeros(x.shape)
            x[x == 0] = 99999.9
            argmin_ = torch.argmin(x, dim=0)
            x = nn.functional.one_hot(argmin_, nns.N)
            return x



class NormalizedPwrSoftmin(nn.Module):
    def forward(self, x):
        x = x[nns.N:]
        x[x == 0] = 9999999999.9
        argmin_ = torch.argmin(x, dim=0)
        x = nn.functional.one_hot(argmin_, nns.N)
        return x
class Simple_Phase_DNN(nn.Module):
    def __init__(self, Inputs, Connectivity_Vector, bias_weights, safedict, ID, init_bias_flag=True):
        super(Simple_Phase_DNN, self).__init__()
        self.steady_coefs = torch.softmax(torch.randn(Inputs//2), dim=0)
        self.ID = ID
        self.safedict = safedict
        self.Connectivity_Vector = Connectivity_Vector
        self.Inputs_ = Inputs
        self.Permafrost_parameters = []
        self.Wrapping_Layers_Input = nn.Linear(Inputs, Inputs, bias=False)
        self.Wrapping_Layers_Input.weight = torch.nn.Parameter(torch.diag(
            torch.cat((Connectivity_Vector, Connectivity_Vector), dim=0)))
        self.Wrapping_Layers_Input.weight.requires_grad = False
        self.Permafrost_parameters.append(self.Wrapping_Layers_Input.weight)
        self.NormalizationLayer = NormalizationLayer()
        self.Wrapping_Layers_Output = nn.Linear(
            Inputs // 2, Inputs // 2, bias=False)
        self.Wrapping_Layers_Output.weight = torch.nn.Parameter(
            torch.diag(Connectivity_Vector))
        self.Wrapping_Layers_Output.weight.requires_grad = False
        self.Permafrost_parameters.append(self.Wrapping_Layers_Output.weight)
        self.bias_layer = nn.Linear(Inputs // 2, Inputs // 2, bias=True)
        self.bias_layer.weight = self.Wrapping_Layers_Output.weight
        if init_bias_flag:
            self.bias_layer.bias = torch.nn.Parameter(bias_weights)
        self.bias_layer.bias.requires_grad = True
        self.bias_layer.weight.requires_grad = False
        self.Permafrost_parameters.append(self.bias_layer.weight)
        self.learning_weighted_averager = Weighted_Averaging_Learner()
        self.LinLayer1 = nn.Linear(self.Inputs_, 30,        bias=False)
        self.LinLayer2 = nn.Linear(30, 30,                  bias=False)
        self.LinLayer3 = nn.Linear(30, self.Inputs_ // 2,   bias=False)
        self.normsoftmax = NormalizedSoftmax()
        self.normsoftmin = NormalizedSoftmin()
        self.LinLayer50 = nn.Linear(self.Inputs_ // 2, self.Inputs_ // 2)
        self.had_layer = HadamardLayer(self.Inputs_ // 2)
        self.normsoftpwrmin = NormalizedPwrSoftmin()

    def switched_layer(self, x):
        norm_factor = 1
        if self.layer_mode == "min_max_avg_neutralize":
            coefs = self.normsoftmax(x, "neutralize") + self.normsoftmin(x, "neutralize")
            norm_factor = coefs.sum() / 2
        if self.layer_mode == "min_max_avg_no_neutralize":
            coefs = self.normsoftmax(x, "no_neutralize") + self.normsoftmin(x, "no_neutralize")
            norm_factor = coefs.sum() / 2
        elif self.layer_mode == "simeone":
            coefs = torch.tensor(self.safedict["simeone_alphas"][:, self.ID])
            pass
        elif self.layer_mode == "random_unsteady":
            coefs = torch.softmax(torch.randn(self.Inputs_//2), dim=0)
            pass
        elif self.layer_mode == "random_steady":
            coefs = self.steady_coefs
            pass
        elif self.layer_mode == "average_over_connections":
            coefs = torch.tensor(self.safedict["con_graph"][:, self.ID] / self.safedict["con_graph"][:, self.ID].sum())
            pass
        return coefs, norm_factor
    def forward(self, x):
        x = self.Wrapping_Layers_Input(x)
        x[:nns.N] = x[:nns.N] #/ torch.max(x[:nns.N].abs())
        x, norm_factor = self.switched_layer(x)
        x = self.NormalizationLayer(x)
        x = self.Wrapping_Layers_Output(x)
        x = self.NormalizationLayer(x) * norm_factor
        return x

class Simple_Period_DNN(nn.Module):
        def __init__(self, Inputs, Connectivity_Vector, bias_weights, safedict, ID, init_bias_flag=True):
            super(Simple_Period_DNN, self).__init__()
            self.ID = ID
            self.steady_coefs = torch.softmax(torch.randn(Inputs//2), dim=0)
            self.safedict = safedict
            self.Inputs_ = Inputs
            self.Permafrost_parameters = []
            self.Connectivity_Vector = Connectivity_Vector
            self.Wrapping_Layers_Input = nn.Linear(Inputs, Inputs, bias=False)
            self.Wrapping_Layers_Input.weight = torch.nn.Parameter(torch.diag(
                torch.cat((Connectivity_Vector, Connectivity_Vector), dim=0)))
            self.Wrapping_Layers_Input.weight.requires_grad = False
            self.Permafrost_parameters.append(self.Wrapping_Layers_Input.weight)
            self.NormalizationLayer = NormalizationLayer()
            self.Wrapping_Layers_Output = nn.Linear(
                Inputs // 2, Inputs // 2, bias=False)
            self.Wrapping_Layers_Output.weight = torch.nn.Parameter(
                torch.diag(Connectivity_Vector))
            self.Wrapping_Layers_Output.weight.requires_grad = False
            self.Permafrost_parameters.append(self.Wrapping_Layers_Output.weight)
            self.bias_layer = nn.Linear(Inputs // 2, Inputs // 2, bias=True)
            self.bias_layer.weight = self.Wrapping_Layers_Output.weight
            if init_bias_flag:
                self.bias_layer.bias = torch.nn.Parameter(bias_weights)
            self.bias_layer.bias.requires_grad = True
            self.bias_layer.weight.requires_grad = False
            self.Permafrost_parameters.append(self.bias_layer.weight)
            self.LinearLayer3 = nn.Linear(self.Inputs_, self.Inputs_ // 2, bias=False)
            self.learning_weighted_averager = Weighted_Averaging_Learner()
            self.LinLayer1 = nn.Linear(self.Inputs_, 30, bias=False)
            self.LinLayer2 = nn.Linear(30, 30, bias=False)
            self.LinLayer3 = nn.Linear(30, self.Inputs_ // 2, bias=False)
            self.LinLayer4 = nn.Linear( Inputs // 2, Inputs // 2, bias=False)
            self.LinLayer4.weight = torch.nn.Parameter(torch.nn.Parameter(torch.diag(Connectivity_Vector)))
            self.internal_eps = nn.Linear(self.Inputs_ // 2, 1)
            self.normsoftmax = NormalizedSoftmax()
            self.normsoftmin = NormalizedSoftmin()
            self.droplayer = torch.nn.Dropout(0.5)
            self.LinLayer50 = nn.Linear(self.Inputs_ // 2, self.Inputs_ // 2)
            self.had_layer = HadamardLayer(self.Inputs_ // 2)
            self.normsoftpwrmin = NormalizedPwrSoftmin()


        def switched_layer(self, x):
            norm_factor = 1
            if self.layer_mode == "min_max_avg_neutralize":
                coefs = self.normsoftmax(x, "neutralize") + self.normsoftmin(x, "neutralize")
                norm_factor = coefs.sum() / 2
            if self.layer_mode == "min_max_avg_no_neutralize":
                coefs = self.normsoftmax(x, "no_neutralize") + self.normsoftmin(x, "no_neutralize")
                norm_factor = coefs.sum() / 2
            elif self.layer_mode == "simeone":
                coefs = torch.tensor(self.safedict["simeone_alphas"][:, self.ID])
                pass
            elif self.layer_mode == "random_unsteady":
                coefs = torch.softmax(torch.randn(self.Inputs_ // 2), dim=0)
                pass
            elif self.layer_mode == "random_steady":
                coefs = self.steady_coefs
                pass
            elif self.layer_mode == "average_over_connections":
                coefs = torch.tensor(
                    self.safedict["con_graph"][:, self.ID] / self.safedict["con_graph"][:, self.ID].sum())
                pass

            return coefs, norm_factor

        def forward(self, x):
            x = self.Wrapping_Layers_Input(x)
            x[:nns.N] = x[:nns.N]  
            x, norm_factor = self.switched_layer(x)
            x = self.NormalizationLayer(x)
            x = self.Wrapping_Layers_Output(x)
            x = self.NormalizationLayer(x) * norm_factor
            return x
      