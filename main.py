# -*- coding: utf-8 -*-

model_name = "train.model"
save_img_path1 = "./saveImg/saveImg1"
save_img_path2 = "./saveImg/saveImg2"
save_img_path3 = "./saveImg/saveImg3"
save_img_path4 = "./saveImg/saveImg4"
save_img_path5 = "./saveImg/saveImg5"


import numpy as np
import os
os.chdir("ml4physim_startingkit_powergrid")
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
sys.path.append('../ml4physim_startingkit_powergrid')
print(sys.path)
import warnings
warnings.filterwarnings('ignore')
import numpy as np
print("-1.1.numpy")
import torch
import pathlib
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import copy
import math
import time
import utils.data_prepare as dp
from lips.dataset.scaler.powergrid_scaler import PowerGridScaler
from lips.dataset.scaler.scaler import Scaler
from lips.dataset.dataSet import DataSet
from lips.dataset.powergridDataSet import PowerGridDataSet
from typing import Union
from matplotlib import pyplot as plt
from lips.benchmark.powergridBenchmark import PowerGridBenchmark
from lips.evaluation.powergrid_evaluation import PowerGridEvaluation
from pprint import pprint
from utils.compute_score import compute_global_score
import warnings


class MyLoss(nn.Module):
	"""
    Custom Loss Function for Power Grid Dispatching

    This class defines a custom loss function tailored for power grid dispatching tasks. It combines multiple constraints
    and loss terms to ensure the predictions meet specific operational requirements of the power grid.

    Attributes:
        None

    Methods:
        forward(self, predict, target, data, YBus, obs, ifPrint=False):
            Computes the custom loss based on the predicted and target values, along with additional data and constraints.

            Parameters:
                predict (torch.Tensor): Predicted values from the model.
                target (torch.Tensor): Target values for comparison.
                data (torch.Tensor): Input data used for additional constraints.
                YBus (torch.Tensor): Admittance matrix of the power grid.
                obs (object): Observation object containing additional grid information.
                ifPrint (bool, optional): Flag to print intermediate results. Defaults to False.

            Returns:
                loss (torch.Tensor): The computed loss value.
                loss_components (dict): Dictionary containing individual loss components.
    """
	def __init__(self):
		super(MyLoss, self).__init__()
		

	def forward(self, predict, target, data, YBus, obs, ifPrint=False):

		# split the predict data
		a_or_pred = predict[:,:186]
		a_ex_pred = predict[:,186:372]
		p_or_pred = predict[:,372:558]
		p_ex_pred = predict[:,558:744]
		v_or_pred = predict[:,744:930]
		v_ex_pred = predict[:,930:]

		# split the target data
		a_or_targ = target[:,:186]
		a_ex_targ = target[:,186:372]
		p_or_targ = target[:,372:558]
		p_ex_targ = target[:,558:744]
		v_or_targ = target[:,744:930]
		v_ex_targ = target[:,930:]

		# split the input data
		prod_p_data = data[:,:62]
		prod_v_data = data[:,62:124]
		load_p_data = data[:,124:223]
		load_q_data = data[:,223:322]
		line_status_data = data[:,322:508]
		topo_vect_data = data[:,508:]


		YBus_edge_indices = YBus[:,0:2,:].int()
		YBus_edge_weights = YBus[:,2,:]


		# Process mutual admittance and obtain resistance which is abandon in the newest version.
        # The RBus can be obtained by the following code.
        # z_pu = [el.r_pu for el in env.backend._grid.get_lines()]
		# z_pu_trafo = [el.r_pu for el in env.backend._grid.get_trafos()]
		# n_lines = len(z_pu)
		# n_trafo = len(z_pu_trafo)
		# z_base = np.power(env.backend.lines_or_pu_to_kv, 2) / env.backend._grid.get_sn_mva()
		# r_ohm = z_pu * z_base[:n_lines]
		# r_ohm = torch.tensor(r_ohm).to(device)

		YBus_not_equal = YBus_edge_indices[:,0] != YBus_edge_indices[:,1]
		YBus_edge_weights_handled = torch.where(YBus_not_equal,-YBus_edge_weights,YBus_edge_weights)
		RBus_edge = torch.where(YBus_not_equal,1/YBus_edge_weights_handled,0).float()

		# MSE
		MSE = torch.mean(torch.pow((predict-target),2))

		# Implement P1 constraint
		condition_1_1 = a_or_pred<0
		condition_1_2 = a_ex_pred<0
		condition_1 = torch.logical_or(condition_1_1,condition_1_2)
		a_or_ex_pred_neg = torch.where(condition_1,torch.ones_like(a_or_pred),torch.zeros_like(a_or_pred))
		P1 = torch.mean(a_or_ex_pred_neg)

		# Implement P2 constraint
		condition_2_1 = v_or_pred<0
		condition_2_2 = v_ex_pred<0
		condition_2 = torch.logical_or(condition_2_1,condition_2_2)
		v_or_ex_pred_neg = torch.where(condition_2,torch.ones_like(v_or_pred),torch.zeros_like(v_or_pred))
		P2 = torch.mean(v_or_ex_pred_neg)

		# Implement P3 constraint
		p_or_ex_pred_sum = p_or_pred + p_ex_pred
		p_or_ex_pred_sum_neg = torch.where(p_or_ex_pred_sum<0,torch.ones_like(p_or_ex_pred_sum),torch.zeros_like(p_or_ex_pred_sum))
		P3 = torch.mean(p_or_ex_pred_sum_neg)

		# Implement P4 constraint
		disconnect_line = torch.nonzero(line_status_data == 1)
		rows = disconnect_line[:,0]
		cols = disconnect_line[:,1]
		a_or_ex_pred_abs_sum = torch.abs(a_or_pred[rows,cols]) + torch.abs(a_ex_pred[rows,cols])
		p_or_ex_pred_abs_sum = torch.abs(p_or_pred[rows,cols]) + torch.abs(p_ex_pred[rows,cols])


		P4 = torch.mean(a_or_ex_pred_abs_sum+p_or_ex_pred_abs_sum).float()

		# Implement P5 constraint
		p_or_ex_pred_dim1sum = (p_ex_pred+p_or_pred).sum(dim=1)
		prod_p_data_dim1sum = prod_p_data.sum(dim=1)
		energy_loss = p_or_ex_pred_dim1sum / prod_p_data_dim1sum
        
		condition_5_1 = energy_loss < 0.005
		condition_5_2 = energy_loss > 0
		condition_5 = torch.logical_and(condition_5_1, condition_5_2)
		energy_loss1 = torch.where(energy_loss>0.04, 200 * (energy_loss-0.04), 0)
		energy_loss2 = torch.where(condition_5, 200 * (0.005 - energy_loss), 0)
		energy_loss3 = torch.where(energy_loss<0, 500 * (0.005 - energy_loss), 0)
		energy_loss = energy_loss1 + energy_loss2 + energy_loss3
		P5 = torch.mean(energy_loss)


		# Implement P6 and P7 constraint
        # We implemented P6 and P7 with KKT-hPINN
		P6=torch.tensor(0)
		P7=torch.tensor(0)

		# Implement P8 constraint
		batch_size = data.shape[0]

		gen_to_subid = obs.gen_to_subid
		load_to_subid = obs.load_to_subid
		line_or_to_subid = obs.line_or_to_subid
		line_ex_to_subid = obs.line_ex_to_subid

		gen_to_subid = torch.tensor(gen_to_subid).to(device)
		load_to_subid = torch.tensor(load_to_subid).to(device)
		line_or_to_subid = torch.tensor(line_or_to_subid).to(device)
		line_ex_to_subid = torch.tensor(line_ex_to_subid).to(device)

		line_s_to_l_sub = torch.concatenate((line_or_to_subid,line_ex_to_subid)).reshape(2,186)
		comparison = line_s_to_l_sub[0] < line_s_to_l_sub[1]
		line_s_to_l_sub_backup = line_s_to_l_sub
		line_s_to_l_sub[0] = torch.where(comparison, line_s_to_l_sub_backup[0], line_s_to_l_sub_backup[1])
		line_s_to_l_sub[1] = torch.where(comparison, line_s_to_l_sub_backup[1], line_s_to_l_sub_backup[0])
		line_s_to_l_sub = torch.tensor(line_s_to_l_sub)

		result_indices = torch.full((batch_size, 186), -1, dtype=torch.int)  # Padding with -1 means not found
		p_or_ex_pred_dim1sum = (p_ex_pred+p_or_pred).sum(dim=1)
		a_or_ex_pred = (a_or_pred+a_ex_pred)*0.5
		line_s_to_l_sub_expanded = line_s_to_l_sub.unsqueeze(0).expand(batch_size, -1, -1)
		line_s_to_l_sub_repeated = line_s_to_l_sub_expanded.unsqueeze(-1).expand(-1, -1, -1, 478)
		
		matches = (line_s_to_l_sub_repeated == YBus_edge_indices.unsqueeze(-2)).all(dim=1)

		# Converts a Boolean type tensor to an integer type
		matches_int = matches.to(torch.int64)

		# Find the first matching index and return -1 if there is no match
		result_indices = matches_int.argmax(dim=-1)

		# If no match is found, the result is set to -1
		no_match_mask = (result_indices == 0) & (~matches.any(dim=-1))
		result_indices[no_match_mask] = -1

		# Creates a tensor that contains the batch index
		batch_indices = torch.arange(batch_size).unsqueeze(1).expand_as(result_indices)

		# Select the corresponding value from RBus_edge using the index tensor
		RBus_selected = RBus_edge[batch_indices, result_indices]

		P81 = (p_or_pred+p_ex_pred)*17.498
		P82 = (torch.mul(torch.pow(a_or_ex_pred,2),RBus_selected))
		P8 = torch.abs(torch.mean(P81 - P82))

		loss = 100*MSE+P1+P2+P3+P4+P5+P6+P7+0.1*P8


		return loss, {'MSE': MSE, 'P1': P1, 'P2': P2, 'P3': P3, 'P4': P4, 'P5': P5, 'P6': P6, 'P7': P7, 'P8': P8}


# The data is normalized
def process_dataset(dataset: DataSet, 
                    scaler: Union[Scaler, None] = None,
                    training: bool=False,
                    ) -> tuple:
        if training:
            inputs, outputs = dataset.extract_data(concat=True)
            
            if scaler is not None:
                inputs, outputs = scaler.fit_transform(dataset)
        else:
            inputs, outputs = dataset.extract_data(concat=True)
            if scaler is not None:
                inputs, outputs = scaler.transform(dataset)
        
        return inputs, outputs

# Generate a batch dataloader
def process_dataloader(inputs,outputs,
                       batch_size: int=128,
                       shuffle: bool=False,
                       YBus=None,
                       dtype1=torch.float32,
                       dtype2=torch.complex128):
    """
    Process and Prepare Data for DataLoader

    This function processes the input and output data, concatenates them as needed, and prepares a PyTorch DataLoader
    for training or evaluation. It can optionally include the YBus matrix if provided.

    Parameters:
        inputs (list of numpy.ndarray): List of input arrays. Each array represents a different feature set.
        outputs (list of numpy.ndarray): List of output arrays. Each array represents a different target set.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 128.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
        YBus (numpy.ndarray, optional): Admittance matrix of the power grid. Defaults to None.
        dtype1 (torch.dtype, optional): Data type for the input and output tensors. Defaults to torch.float32.
        dtype2 (torch.dtype, optional): Data type for the YBus tensor. Defaults to torch.complex128.

    Returns:
        data_loader (DataLoader): PyTorch DataLoader containing the processed data.
    """
    inputs = np.concatenate([inputs[0][0],inputs[0][1],inputs[0][2],
                      inputs[0][3],inputs[1][0],inputs[1][1]],axis=1)
    
    outputs = np.concatenate([outputs[0],outputs[1],outputs[2],
                      outputs[3],outputs[4],outputs[5]],axis=1)
    if YBus is not None:
        torch_dataset = TensorDataset(torch.tensor(inputs, dtype=dtype1), 
                                      torch.tensor(YBus, dtype=dtype2),
                                      torch.tensor(outputs, dtype=dtype1))
        data_loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle,pin_memory=True)
        return data_loader
    
    torch_dataset = TensorDataset(torch.tensor(inputs, dtype=dtype1), 
                                  torch.tensor(outputs, dtype=dtype1))
    data_loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle,pin_memory=True)
    return data_loader


class ProcessingInput(nn.Module):
    def __init__(self, input_size, middle_size) -> None:
        super(ProcessingInput,self).__init__()
        self.middle_size = middle_size
        self.input_size = input_size
        
    def build_model(self):
        self.linear = nn.Linear(self.input_size, self.middle_size)
        self.leakyRelu = nn.LeakyReLU()
        
    def forward(self, data):

        data = self.linear(data)
        data = self.leakyRelu(data)
        return data

class ResNetLayer(nn.Module):
    def __init__(self,
                 middle_size,
                 activation = nn.LeakyReLU):
        super(ResNetLayer,self).__init__()
        self.middle_size = middle_size
        self.d = None
        self.e = None
        self.activation = activation()
    def build_model(self):
        self.e = nn.Linear(self.middle_size, self.middle_size)
        self.d = nn.Linear(self.middle_size, self.middle_size)
    
    def forward(self, x):
        res = self.e(x)
        res = self.activation(res)
        res = self.d(res)
        res = self.activation(res)
        res = x+res
        
        return x
        


class LtauNoAdd(nn.Module):
    def __init__(self,middle_size,tau_size):
        super(LtauNoAdd,self).__init__()
        self.middle_size = middle_size
        self.tau_size = tau_size
        self.e = None
        self.d = None
    
    def build_model(self):
        self.e = nn.Linear(self.middle_size,self.tau_size)
        self.d = nn.Linear(self.tau_size,self.middle_size)

    def forward(self,x,tau):
        tmp = self.e(x)
        tmp = torch.mul(tmp,tau)
        res = self.d(tmp)
        return res
    
class DecoderLayer(nn.Module):
    def __init__(self,middle_size,output_size,activation=nn.LeakyReLU):
        super(DecoderLayer,self).__init__()
        self.middle_size = middle_size
        self.output_size = output_size
        
        self.d1 = None
        self.d2 = None
        self.activation = activation()
    def build_model(self):
        self.d1 = nn.Linear(self.middle_size,self.middle_size)
        self.d2 = nn.Linear(self.middle_size,self.output_size)
        
        
    def forward(self,x):
        res = self.d1(x)
        res = self.activation(res)
        res = self.d2(res)
        
        return res
    

class UnscalingLayer(nn.Module):
    def __init__(self,powerGridScaler,device):
        super(UnscalingLayer, self).__init__()
        self.powerGridScaler = powerGridScaler
        self._m_x,self._sd_x,self._m_y,self._sd_y,self._m_tau,self._sd_tau = powerGridScaler.get_all_m_sd()

    def build_model(self):
        self._m_x = np.concatenate(self._m_x, axis=0)
        if(type(self._sd_x[1])== float):
            self._sd_x[1] = np.full(62, self._sd_x[1])
        
        self._m_y_tmp = []
        self._sd_x = np.concatenate(self._sd_x, axis=0)
        if(len(self._m_y) == 6):
            self._m_y_tmp.append(np.full(186, self._m_y[0]))
            self._m_y_tmp.append(np.full(186, self._m_y[1])) 
            self._m_y_tmp.append(np.full(186, self._m_y[2]))
            self._m_y_tmp.append(np.full(186, self._m_y[3]))
            self._m_y_tmp.append(np.full(186, self._m_y[4]))
            self._m_y_tmp.append(np.full(186, self._m_y[5]))
            self._m_y = np.concatenate(self._m_y_tmp, axis=0)
        else:
            self._m_y = np.concatenate(self._m_y, axis=0)
        self._sd_y = np.concatenate(self._sd_y, axis=0)
        
        self.device = device

        self._m_x = torch.tensor(self._m_x).to(self.device)
        self._sd_x = torch.tensor(self._sd_x).to(self.device)
        self._m_y = torch.tensor(self._m_y).to(self.device)
        self._sd_y = torch.tensor(self._sd_y).to(self.device)
        self._m_tau = torch.tensor(self._m_tau).to(self.device)
        self._sd_tau = torch.tensor(self._sd_tau).to(self.device)

    def forward(self,x,y,tau):
        # x => (128,322) = 62+62+99+99
        # y => (128,1116) = 186*6
        # tau => (128,726)
        x = x*self._sd_x + self._m_x
        y = y*self._sd_y + self._m_y

        # tau = tau*self._sd_tau + self._m_tau
        return x,y,tau
        
    def show(self):
        print(f"m_x:{self._m_x}")
        print(f"sd_x:{self._sd_x}")
        print(f"m_y:{self._m_y}")
        print(f"sd_y:{self._sd_y}")
        print(f"m_tau:{self._m_tau}")
        print(f"sd_tau:{self._sd_tau}")
    
    def show_shape(self):
        print(f"m_x:{self._m_x.shape}")
        print(f"sd_x:{self._sd_x.shape}")
        print(f"m_y:{self._m_y.shape}")
        print(f"sd_y:{self._sd_y.shape}")
        print(f"m_tau:{self._m_tau}")
        print(f"sd_tau:{self._sd_tau}")



def KKTP7ABb(obs,device):

    sub_number = 118
    line_number = 186

    line_or_to_subid = torch.tensor(obs.line_or_to_subid).to(device)
    line_ex_to_subid = torch.tensor(obs.line_ex_to_subid).to(device)

    
    B1 = torch.zeros((line_number,sub_number))
    B2 = torch.zeros((line_number,sub_number))
    B1[torch.arange(line_number), line_or_to_subid] = 1
    B2[torch.arange(line_number), line_ex_to_subid] = 1
    B1 = B1.T
    B2 = B2.T
    
    A = torch.eye(sub_number)
    B = -torch.concatenate((B1,B2),axis=1)
    b = torch.zeros(sub_number)

    BBTinversed = torch.linalg.inv(B @ B.T)

    A_star = - B.T @ BBTinversed @ A
    B_star = torch.eye(line_number*2) - B.T @ BBTinversed @ B
    b_star = B.T @ BBTinversed @ b

    
    A_star = A_star.to(device)
    B_star = B_star.to(device)
    b_star = b_star.to(device).unsqueeze(1)


    return A_star,B_star,b_star
def KKTP7Xp(obs,device,data):
    batch_size = data.shape[0]
    sub_number = 118
    line_number = 186
    prod_p_data = data[:,:62]
    load_p_data = data[:,124:223]
    gen_to_subid = torch.tensor(obs.gen_to_subid).to(device)
    load_to_subid = torch.tensor(obs.load_to_subid).to(device)
    gen_power_sums = torch.zeros((sub_number, batch_size)).to(device)
    load_power_sums = torch.zeros((sub_number, batch_size)).to(device)
    # This is equivalent to copying gen_to_subid[62] 118 times into [118,62], each row being gen_to_subid, and then converting (0,118) to arange's [118,1] matrix
    # Go up the column, comparing [0:118,0] and [118,1] for gen_to_subid, then [0:118,1] and [118,1] for gen_to_subid, and so on, comparing all 62 columns
    gen_to_subid_one_hot = (gen_to_subid.unsqueeze(0) == torch.arange(sub_number).unsqueeze(1).to(device)).float()
    gen_power_sums = torch.matmul(gen_to_subid_one_hot, prod_p_data.T.float())
    load_to_subid_one_hot = (load_to_subid.unsqueeze(0) == torch.arange(sub_number).unsqueeze(1).to(device)).float()
    load_power_sums = torch.matmul(load_to_subid_one_hot, load_p_data.T.float())
    
    X_p = (gen_power_sums-load_power_sums)
    return X_p

def KKTP4(data_tau,output_inversed):
    line_status = data_tau[:,0:186]
    disconnect_line = torch.nonzero(line_status == 1)
    rows = disconnect_line[:,0]
    cols = disconnect_line[:,1]
    output_inversed[:,:186][rows,cols] = 0
    output_inversed[:,186:372][rows,cols] = 0
    output_inversed[:,372:558][rows,cols] = 0
    output_inversed[:,558:744][rows,cols] = 0
    output_inversed[:,744:930][rows,cols] = 0
    output_inversed[:,930:1116][rows,cols] = 0
    return output_inversed





from lips.augmented_simulators import AugmentedSimulator

class LEAPNet(AugmentedSimulator):
    def __init__(self,
             processingInput,
             resNetLayer1,
             resNetLayer2,
             ltauNoAdd,
             decoderLayer1,
             decoderLayer2,
             decoderLayer3,
             decoderLayer4,
             decoderLayer5,
             decoderLayer6,
             unscalingLayer,
             powerGridScaler,
             device,
             obs
             ):
        
        super(LEAPNet, self).__init__()
        self.mse_list = []
        self.P1 = []
        self.P2 = []
        self.P3 = []
        self.P4 = []
        self.P5 = []
        self.P6 = []
        self.P7 = []
        self.P8 = []
        self.val_losses = []
        self.processingInput = processingInput
        self.resNetLayer1 = resNetLayer1
        self.resNetLayer2 = resNetLayer2
        self.ltauNoAdd = ltauNoAdd
        self.decoderLayer1 = decoderLayer1
        self.decoderLayer2 = decoderLayer2
        self.decoderLayer3 = decoderLayer3
        self.decoderLayer4 = decoderLayer4
        self.decoderLayer5 = decoderLayer5
        self.decoderLayer6 = decoderLayer6

        self.unscalingLayer = unscalingLayer
        self.powerGridScaler = powerGridScaler
        self.device = device
        self.obs = obs

    def build_model(self):
        self.processingInput.build_model()
        self.resNetLayer1.build_model()
        self.resNetLayer2.build_model()
        self.ltauNoAdd.build_model()
        self.decoderLayer1.build_model()
        self.decoderLayer2.build_model()
        self.decoderLayer3.build_model()
        self.decoderLayer4.build_model()
        self.decoderLayer5.build_model()
        self.decoderLayer6.build_model()
        self.unscalingLayer.build_model()

    def train(self,train_loader,val_loader,
              epochs=100,lr = 3e-4,start=True):
        if start:
            self.build_model()
        self.trained = True
        
        self.processingInput.to(self.device)
        self.resNetLayer1.to(self.device)
        self.resNetLayer2.to(self.device)
        self.ltauNoAdd.to(self.device)
        self.decoderLayer1.to(self.device)
        self.decoderLayer2.to(self.device)
        self.decoderLayer3.to(self.device)
        self.decoderLayer4.to(self.device)
        self.decoderLayer5.to(self.device)
        self.decoderLayer6.to(self.device)
        self.unscalingLayer.to(self.device)
        
        
        optimizer = torch.optim.Adam([
            {'params':self.processingInput.parameters()},
            {'params':self.resNetLayer1.parameters()},
            {'params':self.resNetLayer2.parameters()},
            {'params':self.ltauNoAdd.parameters()},
            {'params':self.decoderLayer1.parameters()},
            {'params':self.decoderLayer2.parameters()},
            {'params':self.decoderLayer3.parameters()},
            {'params':self.decoderLayer4.parameters()},
            {'params':self.decoderLayer5.parameters()},
            {'params':self.decoderLayer6.parameters()},
            {'params':self.unscalingLayer.parameters()}],
            lr=lr
        )
        loss_function = MyLoss()
        self.P7A_star,self.P7B_star,self.P7b_star = KKTP7ABb(self.obs,self.device)
        for epoch in range(epochs):
            
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
            self.processingInput.train()
            self.resNetLayer1.train()
            self.resNetLayer2.train()
            self.ltauNoAdd.train()
            self.decoderLayer1.train()
            self.decoderLayer2.train()
            self.decoderLayer3.train()
            self.decoderLayer4.train()
            self.decoderLayer5.train()
            self.decoderLayer6.train()
            self.unscalingLayer.train()
            total_loss = 0
            for batch in train_loader:
                # The data is going to be used in the loss, hold it, and use the output as the result
                data,YBus,target = batch
                data_tau = data[:,322:]
                data = data[:,0:322]
                data_tau = data_tau.to(self.device)
                data = data.to(self.device)
                YBus = YBus.to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                output = self.processingInput(data)
                output = self.resNetLayer1(output)
                output = self.resNetLayer2(output)
                output_mid = self.ltauNoAdd(output,data_tau)
                output = output + output_mid
                a_or_pred = self.decoderLayer1(output)
                a_ex_pred = self.decoderLayer2(output)
                p_or_pred = self.decoderLayer3(output)
                p_ex_pred = self.decoderLayer4(output)
                v_or_pred = self.decoderLayer5(output)
                v_ex_pred = self.decoderLayer6(output)
                output = torch.cat((a_or_pred,a_ex_pred,p_or_pred,p_ex_pred,v_or_pred,v_ex_pred),dim=1)
                data_inversed,output_inversed,data_tau = self.unscalingLayer(data,output,data_tau)
                _,target_inversed,_ = self.unscalingLayer(data,target,data_tau)
                # Zero out the disconnected line voltage, current, and power
                output_inversed = KKTP4(data_tau,output_inversed)
                # P7 KKT-hPINN
                P7X = KKTP7Xp(self.obs,self.device,data_inversed)
                p_tmp = output_inversed[:,372:744].T
                output_inversed[:,372:744] = (self.P7A_star @ P7X + self.P7B_star @ p_tmp + self.P7b_star).T
                data_inversed = torch.cat((data_inversed,data_tau),dim=1)
                loss,extra_vars = loss_function(output_inversed,target_inversed,data_inversed,YBus,self.obs)

                loss.backward()
                optimizer.step()
                total_loss += loss.item() *len(data)
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
            mean_loss = total_loss / len(train_loader.dataset)
            print(f'Train Epoch: {epoch}, Avg_Loss: {mean_loss:.5f}')
            self.train_losses.append(mean_loss)
            self.mse_list.append(extra_vars['MSE'].item())
            self.P1.append(extra_vars['P1'].item())
            self.P2.append(extra_vars['P2'].item())
            self.P3.append(extra_vars['P3'].item())
            self.P4.append(extra_vars['P4'].item())
            self.P5.append(extra_vars['P5'].item())
            self.P6.append(extra_vars['P6'].item())
            self.P7.append(extra_vars['P7'].item())
            self.P8.append(extra_vars['P8'].item())
            print(extra_vars)
        return self.train_losses,self.val_losses
    def validate(self,val_loader):
        self.processingInput.eval()
        self.resNetLayer1.eval()
        self.resNetLayer2.eval()
        self.ltauNoAdd.eval()
        self.decoderLayer1.eval()
        self.decoderLayer2.eval()
        self.decoderLayer3.eval()
        self.decoderLayer4.eval()
        self.decoderLayer5.eval()
        self.decoderLayer6.eval()
        self.unscalingLayer.eval()
        total_loss = 0
        loss_function = MyLoss()
        self.P7A_star,self.P7B_star,self.P7b_star = KKTP7ABb(self.obs,self.device)
        
        with torch.no_grad():
            for batch in val_loader:
                data,YBus,target = batch
                data_tau = data[:,322:]
                data = data[:,0:322]
                data_tau = data_tau.to(self.device)
                data = data.to(self.device)
                YBus = YBus.to(self.device)
                target = target.to(self.device)
                output = self.processingInput(data)
                output = self.resNetLayer1(output)
                output = self.resNetLayer2(output)
                output_mid = self.ltauNoAdd(output,data_tau)
                output = output + output_mid
                a_or_pred = self.decoderLayer1(output)
                a_ex_pred = self.decoderLayer2(output)
                p_or_pred = self.decoderLayer3(output)
                p_ex_pred = self.decoderLayer4(output)
                v_or_pred = self.decoderLayer5(output)
                v_ex_pred = self.decoderLayer6(output)
                output = torch.cat((a_or_pred,a_ex_pred,p_or_pred,p_ex_pred,v_or_pred,v_ex_pred),dim=1)
                data_inversed,output_inversed,data_tau = self.unscalingLayer(data,output,data_tau)
                _,target_inversed,_ = self.unscalingLayer(data,target,data_tau)
                # Zero out the disconnected line voltage, current, and power
                output_inversed = KKTP4(data_tau,output_inversed)
                # P7 KKT-hPINN
                P7X = KKTP7Xp(self.obs,self.device,data_inversed)
                p_tmp = output_inversed[:,372:744].T

                output_inversed[:,372:744] = (self.P7A_star @ P7X + self.P7B_star @ p_tmp + self.P7b_star).T

                

                data_inversed = torch.cat((data_inversed,data_tau),dim=1)
                
                loss,_ = loss_function(output_inversed,target_inversed,data_inversed,YBus,self.obs)
                total_loss += loss.item() *len(data)
            mean_loss = total_loss / len(val_loader.dataset)
            
            print(f"Eval:   Avg_Loss: {mean_loss:.5f}")
        return mean_loss
    def predict(self,dataset,eval_batch_size=128):
        self.processingInput.eval()
        self.resNetLayer1.eval()
        self.resNetLayer2.eval()
        self.ltauNoAdd.eval()
        self.decoderLayer1.eval()
        self.decoderLayer2.eval()
        self.decoderLayer3.eval()
        self.decoderLayer4.eval()
        self.decoderLayer5.eval()
        self.decoderLayer6.eval()
        self.unscalingLayer.eval()
        outputs = []
        targets = []
        inputs_test,outputs_test = process_dataset(dataset = dataset,
                                                   scaler=self.powerGridScaler,
                                                   training=False)
        dataloader_test = process_dataloader(inputs = inputs_test,
                                             outputs = outputs_test,
                                             shuffle=False,
                                             batch_size=eval_batch_size,
                                             )
        
        with torch.no_grad():
            for batch in dataloader_test:
                data,target = batch
                data_tau = data[:,322:]
                data = data[:,0:322]
                data_tau = data_tau.to(self.device)
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.processingInput(data)
                output = self.resNetLayer1(output)
                output = self.resNetLayer2(output)
                output_mid = self.ltauNoAdd(output,data_tau)
                output = output + output_mid
                a_or_pred = self.decoderLayer1(output)
                a_ex_pred = self.decoderLayer2(output)
                p_or_pred = self.decoderLayer3(output)
                p_ex_pred = self.decoderLayer4(output)
                v_or_pred = self.decoderLayer5(output)
                v_ex_pred = self.decoderLayer6(output)
                output = torch.cat((a_or_pred,a_ex_pred,p_or_pred,p_ex_pred,v_or_pred,v_ex_pred),dim=1)
                data_inversed,output,_ = self.unscalingLayer(data,output,data_tau)
                _,target_inversed,_ = self.unscalingLayer(data,target,data_tau)
                output = KKTP4(data_tau,output)
                P7X = KKTP7Xp(self.obs,self.device,data_inversed)
                p_tmp = output[:,372:744].T

                output[:,372:744] = (self.P7A_star @ P7X + self.P7B_star @ p_tmp + self.P7b_star).T
                
                if self.device == torch.device('cpu'):
                    
                    outputs.append(output.numpy())
                    targets.append(target_inversed.numpy())
                else:
                    
                    outputs.append(output.cpu().numpy())
                    targets.append(target_inversed.cpu().numpy())
        outputs = np.concatenate(outputs)
        outputs = dataset.reconstruct_output(outputs)
        targets = np.concatenate(targets)
        targets = dataset.reconstruct_output(targets)

        return outputs,targets
                
    def summary(self):
        """
        summary of the model
        """
        print(self._model)

    def count_parameters(self):
        """
        count the number of parameters in the model
        """
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    def visualize_convergence0(self, figsize=(15,5), save_path: str=None):
        """
        Visualizing the convergence of the model
        """
        # raise an error if the train_losses is empty
        if len(self.train_losses) == 0:
            raise RuntimeError("The model should be trained before visualizing the convergence")
        num_metrics = 1
        if num_metrics == 0:
            nb_subplots = 1
        else:
            nb_subplots = num_metrics + 1
        fig, ax = plt.subplots(1,nb_subplots, figsize=figsize)

        ax[0].set_title("train")
        ax[0].plot(self.train_losses, label='train_loss')

        ax[1].set_title("mse")
        ax[1].plot(self.mse_list, label='mse')

        for i in range(nb_subplots):
            ax[i].grid()
            ax[i].legend()
        # save the figure
        if save_path is not None:
            if not pathlib.Path(save_path).exists():
                pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)

    def visualize_convergence12(self, figsize=(15,5), save_path: str=None):
        """
        Visualizing the convergence of the model
        """
        # raise an error if the train_losses is empty
        if len(self.train_losses) == 0:
            raise RuntimeError("The model should be trained before visualizing the convergence")
        num_metrics = 1
        if num_metrics == 0:
            nb_subplots = 1
        else:
            nb_subplots = num_metrics + 1
        fig, ax = plt.subplots(1,nb_subplots, figsize=figsize)

        ax[0].set_title("P1")
        ax[0].plot(self.P1, label='P1')
        ax[1].set_title("P2")
        ax[1].plot(self.P2, label='P2')

        for i in range(nb_subplots):
            ax[i].grid()
            ax[i].legend()
        # save the figure
        if save_path is not None:
            if not pathlib.Path(save_path).exists():
                pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)

    def visualize_convergence34(self, figsize=(15,5), save_path: str=None):
        """
        Visualizing the convergence of the model
        """
        # raise an error if the train_losses is empty
        if len(self.train_losses) == 0:
            raise RuntimeError("The model should be trained before visualizing the convergence")
        num_metrics = 1
        if num_metrics == 0:
            nb_subplots = 1
        else:
            nb_subplots = num_metrics + 1
        fig, ax = plt.subplots(1,nb_subplots, figsize=figsize)

        ax[0].set_title("P3")
        ax[0].plot(self.P3, label='P3')
        ax[1].set_title("P4")
        ax[1].plot(self.P4, label='P4')

        for i in range(nb_subplots):
            ax[i].grid()
            ax[i].legend()
        # save the figure
        if save_path is not None:
            if not pathlib.Path(save_path).exists():
                pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)

    def visualize_convergence56(self, figsize=(15,5), save_path: str=None):
        """
        Visualizing the convergence of the model
        """
        # raise an error if the train_losses is empty
        if len(self.train_losses) == 0:
            raise RuntimeError("The model should be trained before visualizing the convergence")
        num_metrics = 1
        if num_metrics == 0:
            nb_subplots = 1
        else:
            nb_subplots = num_metrics + 1
        fig, ax = plt.subplots(1,nb_subplots, figsize=figsize)

        ax[0].set_title("P5")
        ax[0].plot(self.P5, label='P5')
        ax[1].set_title("P6")
        ax[1].plot(self.P6, label='P6')

        for i in range(nb_subplots):
            ax[i].grid()
            ax[i].legend()
        # save the figure
        if save_path is not None:
            if not pathlib.Path(save_path).exists():
                pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)

    def visualize_convergence78(self, figsize=(15,5), save_path: str=None):
        """
        Visualizing the convergence of the model
        """
        # raise an error if the train_losses is empty
        if len(self.train_losses) == 0:
            raise RuntimeError("The model should be trained before visualizing the convergence")
        num_metrics = 1
        if num_metrics == 0:
            nb_subplots = 1
        else:
            nb_subplots = num_metrics + 1
        fig, ax = plt.subplots(1,nb_subplots, figsize=figsize)

        ax[0].set_title("P7")
        ax[0].plot(self.P7, label='P7')
        ax[1].set_title("P8")
        ax[1].plot(self.P8, label='P8')

        for i in range(nb_subplots):
            ax[i].grid()
            ax[i].legend()
        # save the figure
        if save_path is not None:
            if not pathlib.Path(save_path).exists():
                pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)

     
from lips.benchmark.powergridBenchmark import get_env, get_kwargs_simulator_scenario

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    # Use some required pathes
    
    DATA_PATH = pathlib.Path().resolve() / "input_data_local" / "lips_idf_2023"
    BENCH_CONFIG_PATH = pathlib.Path().resolve() / "configs" / "benchmarks" / "lips_idf_2023.ini"
    SIM_CONFIG_PATH = pathlib.Path().resolve() / "configs" / "simulators"
    TRAINED_MODELS = pathlib.Path().resolve() / "input_data_local" / "trained_models"
    
    benchmark_kwargs = {"attr_x": ("prod_p", "prod_v", "load_p", "load_q"),
                        "attr_y": ("a_or", "a_ex", "p_or", "p_ex", "v_or", "v_ex"),
                        "attr_tau": ("line_status", "topo_vect"),
                        "attr_physics": ("YBus",)}
    benchmark = PowerGridBenchmark(benchmark_name="Benchmark_competition",
                                benchmark_path=DATA_PATH,
                                load_data_set=True,
                                log_path=None,
                                config_path=BENCH_CONFIG_PATH,
                                load_ybus_as_sparse=True,
                                **benchmark_kwargs
                                )
    
    env_kwargs = get_kwargs_simulator_scenario(benchmark.config)
    env = get_env(env_kwargs)
    obs = env.reset()

    # Handle YBus metrics
    YBus_metric_train = benchmark.train_dataset.data["YBus"]
    YBus_metric_val = benchmark.val_dataset.data["YBus"]
    YBus_train = dp.get_YBus_all_tensor(YBus_metric_train,False)
    YBus_val = dp.get_YBus_all_tensor(YBus_metric_val,False)
    # We changed the old powerridscaler, please overwrite the code in the "new scaler" file to the old powerridscaler.
    powerGridScaler =  PowerGridScaler()
    inputs_train,outputs_train = process_dataset(dataset = benchmark.train_dataset,
                                                scaler = powerGridScaler,
                                                training = True)
    dataloader_train = process_dataloader(inputs = inputs_train,
                                        outputs = outputs_train,
                                        YBus=YBus_train,
                                        shuffle=False)
        
    inputs_val,outputs_val = process_dataset(dataset = benchmark.val_dataset,
                                                scaler = powerGridScaler,
                                                training = False)
    dataloader_val = process_dataloader(inputs = inputs_val,
                                        outputs = outputs_val,
                                        YBus=YBus_val,
                                        shuffle = True)
    
    processingInputInstance = ProcessingInput(322,900)
    resNetLayerInstance1 = ResNetLayer(900)
    resNetLayerInstance2 = ResNetLayer(900)
    ltauNoAddInstance = LtauNoAdd(900,726)
    decoderLayerInstance1 = DecoderLayer(900,186)
    decoderLayerInstance2 = DecoderLayer(900,186)
    decoderLayerInstance3 = DecoderLayer(900,186)
    decoderLayerInstance4 = DecoderLayer(900,186)
    decoderLayerInstance5 = DecoderLayer(900,186)
    decoderLayerInstance6 = DecoderLayer(900,186)
    unscalingLayerInstance = UnscalingLayer(powerGridScaler,device)
    leapNetInstance = LEAPNet(processingInput=processingInputInstance,
                          resNetLayer1=resNetLayerInstance1,
                          resNetLayer2=resNetLayerInstance2,
                          ltauNoAdd=ltauNoAddInstance,
                          decoderLayer1=decoderLayerInstance1,
                          decoderLayer2=decoderLayerInstance2,
                          decoderLayer3=decoderLayerInstance3,
                          decoderLayer4=decoderLayerInstance4,
                          decoderLayer5=decoderLayerInstance5,
                          decoderLayer6=decoderLayerInstance6,
                          unscalingLayer=unscalingLayerInstance,
                          powerGridScaler=powerGridScaler,
                          device=device,
                          obs=obs
                          )
    train_losses,val_losses = leapNetInstance.train(train_loader=dataloader_train,
                                        val_loader=dataloader_val,
                                        epochs=100,lr=3e-4,start=True)
    
    train_losses,val_losses = leapNetInstance.train(train_loader=dataloader_train,
                                        val_loader=dataloader_val,
                                        epochs=100,lr=3e-5,start=False)
    train_losses,val_losses = leapNetInstance.train(train_loader=dataloader_train,
                                        val_loader=dataloader_val,
                                        epochs=50,lr=3e-6,start=False)
    
    leapNetInstance.visualize_convergence0(save_path=save_img_path1)
    leapNetInstance.visualize_convergence12(save_path=save_img_path2)
    leapNetInstance.visualize_convergence34(save_path=save_img_path3)
    leapNetInstance.visualize_convergence56(save_path=save_img_path4)
    leapNetInstance.visualize_convergence78(save_path=save_img_path5)
    predictions,observations = leapNetInstance.predict(benchmark._test_dataset)
    SAVE_PATH = os.path.join(TRAINED_MODELS, benchmark.env_name,model_name)
    torch.save(leapNetInstance,SAVE_PATH)


    env = get_env(get_kwargs_simulator_scenario(benchmark.config))
    evaluator = PowerGridEvaluation(benchmark.config)
    metrics_test = evaluator.evaluate(observations=benchmark._test_dataset.data,
                                    predictions=predictions,
                                    dataset=benchmark._test_dataset,
                                    augmented_simulator=leapNetInstance,
                                    env=env)
    pprint(metrics_test)

    metrics_all = dict()
    metrics_all["test"] = metrics_test
    predictions, observations = leapNetInstance.predict(benchmark._test_ood_topo_dataset)
    evaluator = PowerGridEvaluation(benchmark.config)
    metrics_ood = evaluator.evaluate(observations=benchmark._test_ood_topo_dataset.data,
                                    predictions=predictions,
                                    env=env)
    pprint(metrics_ood)
    metrics_all["test_ood_topo"] = metrics_ood
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        score = compute_global_score(metrics_all, benchmark.config)