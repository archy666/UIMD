import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from typing import List, Callable, Union, Any, TypeVar, Tuple
import random

class MLP_CU(nn.Module):
    def __init__(self, input_size: int, class_num: int, hidden_dims_C: List = None, hidden_dims_U: List = None):
        super(MLP_CU, self).__init__()
        input_C = input_size 
        # Build Encoder_C
        modules_C = []
        if hidden_dims_C is None:
            hidden_dims_C = [ceil(input_size*1.2), ceil(input_size*0.5), ceil(class_num*10)]  
        for h_dim in hidden_dims_C:
                    modules_C.append(
                        nn.Sequential(
                            nn.Linear(input_C,out_features=h_dim),
                            nn.ReLU()
                        )
                    )
                    input_C = h_dim
        self.encoder_C = nn.Sequential(*modules_C)
        
        # Build Encoder_U
        input_U = input_size 
        modules_U = []
        if hidden_dims_U is None:
            hidden_dims_U = [ceil(input_size*1.2), ceil(input_size*0.5), ceil(class_num*10)] #[ceil(input_size*0.52), ceil(input_size*1.2), ceil(class_num*1.2)]
        for h_dim in hidden_dims_U:
            modules_U.append(
                nn.Sequential(
                    nn.Linear(input_U, out_features=h_dim),
                    nn.ReLU()
                )
            )
            input_U = h_dim
        self.encoder_U = nn.Sequential(*modules_U)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, input):
        z_C = self.encoder_C(input)
        z_U = self.encoder_U(input)
        return  [z_C, z_U]
    
class UIMD(nn.Module):
    ''' Model for Incomplete Multiview Compact Information '''
    def __init__(self, input_dims, class_num) -> None:
        super(UIMD, self).__init__()
        self.n_view = len(input_dims)
        self.X_nets = nn.ModuleList([MLP_CU(input_size=input_dims[i], class_num=class_num) for i in range(self.n_view)])
        self.dropout = nn.Dropout(p=0.5)
        classifier_input_dim = ceil(class_num*10) + 1*(ceil(class_num*10))
        self.classifier = nn.Linear(classifier_input_dim, class_num)
        self.weights = torch.nn.Parameter(torch.full((len(input_dims),), 1 / len(input_dims)))


    def forward(self, inputs, Sn_s):
        assert len(inputs) == self.n_view, f"Expected {self.n_view} inputs, but got {len(inputs)}"
        assert len(Sn_s) == self.n_view, f"Expected {self.n_view} Sn_s, but got {len(Sn_s)}"
        com_features, uni_features = [], [] 
        for i, (input_i, Sn_i) in enumerate(zip(inputs, Sn_s)):
            X_net_i = self.X_nets[i]
            com_z_i, uni_zi = X_net_i(input_i)
            com_z_i, uni_zi = com_z_i * Sn_i, uni_zi * Sn_i
            com_features.append(com_z_i)
            uni_features.append(uni_zi) 
        C_list=[random.choice(com_feature_i[Sn_i.squeeze().nonzero()]).squeeze() \
                for com_feature_i, Sn_i in zip(torch.stack(com_features).transpose(0,1), 
                                               torch.stack(Sn_s).transpose(0,1))]
        com_feature=torch.stack(C_list)
        S_weights_sum_to_one = nn.functional.softmax(self.weights, dim=0)
        uni_feature = sum([uni_fea * w for (uni_fea, w) in  zip(uni_features, S_weights_sum_to_one)])
        feature = torch.cat((com_feature, uni_feature), dim=1)
        out    = self.classifier(feature)
        return inputs, com_feature, com_features, uni_features, out
    
