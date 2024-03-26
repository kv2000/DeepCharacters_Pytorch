"""
@File: WootGCN.py
@Author: Heming Zhu
@Email: hezhu@mpi-inf.mpg.de
@Date: 2023-09-25
@Desc: More like a direct translation (without any optimization) from GCN's from the original (Real-time Deep Dynamic Characters. Sigraph2021, Marc Habermann et.al ).
"""

import os
import sys
sys.path.append("../")
sys.path.append("../../")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange

def compute_neighbor_with_ring(st_vert =None, neighbor_idx = None, distance_limit = 1145141919, normalize = None):
    num_verts = len(neighbor_idx)
    q_q = [st_vert]
    ret_dist = (np.ones(num_verts) * 1145141919).astype(np.int32)
    ret_dist[st_vert] = 0
    in_the_queue = np.zeros(num_verts).astype(np.bool_)
    in_the_queue[st_vert] = True

    valid_arr = [st_vert]

    while(len(q_q) > 0):    
        cur_id = q_q[0]
        q_q.pop(0)
         
        for next_id in neighbor_idx[cur_id]:
            if (
                ((ret_dist[cur_id] + 1) < ret_dist[next_id]) and ((ret_dist[cur_id] + 1) < distance_limit )
            ):
                ret_dist[next_id] = ret_dist[cur_id] + 1
                if not in_the_queue[next_id]:
                    q_q.append(next_id)
                    in_the_queue[next_id] = True
                    valid_arr.append(next_id)

        in_the_queue[cur_id] = False
    
    valid_arr = list(set(valid_arr))
    valid_arr.sort()
    valid_arr = np.array(valid_arr)
    selected_dist = np.array(ret_dist)[valid_arr]
    selected_weight = distance_limit - selected_dist

    if normalize:
        selected_weight = selected_weight / np.sum(selected_weight)
    
    return valid_arr, selected_weight

class GCNNOperator(nn.Module):
    def __init__(self, 
            fNew, fOld = None, denseInitializerScale = None, A = None, denseConnect = False, numGraphNodes = 0, 
            sharedWeights='no_sharing'
        ):
        super(GCNNOperator, self).__init__()
        self.fNew                       = fNew
        self.fOld                       = fOld
        self.denseInitializerScale      = denseInitializerScale
        self.A                          = A
        self.denseConnect               = denseConnect
        self.numGraphNodes              = numGraphNodes
        self.sharedWeights              = sharedWeights

        self.kernel = None
        self.kernel_pre = None
        self.bias_pre = None
        self.bias = None
        
        #######################################
        # initalize the kernel

        if self.denseConnect:
            kernel_val = torch.empty([self.numGraphNodes, self.numGraphNodes], requires_grad=True)
            torch.nn.init.normal_(kernel_val, -1, 1)
            kernel_val = kernel_val * self.denseInitializerScale
            self.kernel = torch.nn.Parameter(kernel_val)
        else:
            if self.sharedWeights == 'sharing':
                kernel_val = torch.empty([1, self.fOld, self.fNew], requires_grad=True)
                torch.nn.init.normal_(kernel_val, -1, 1)
                kernel_val = kernel_val * self.denseInitializerScale
                self.kernel_pre = torch.nn.Parameter(kernel_val)
                self.kernel = self.kernel_pre.expand(
                    [self.numGraphNodes, -1, -1]
                )
            else:
                kernel_val = torch.empty([self.numGraphNodes, self.fOld, self.fNew], requires_grad=True)
                torch.nn.init.normal_(kernel_val, -1, 1)
                kernel_val = kernel_val * self.denseInitializerScale
                self.kernel = torch.nn.Parameter(kernel_val)  

        #######################################
        # initalize the bias
        if self.sharedWeights == 'sharing':
            bias_val = torch.empty([1, self.fNew], requires_grad=True)
            torch.nn.init.normal_(bias_val, -1, 1)
            bias_val = bias_val * self.denseInitializerScale
            self.bias_pre = torch.nn.Parameter(bias_val)
            self.bias = self.bias_pre.expand(
                [self.numGraphNodes, -1]
            )
        else:
            bias_val = torch.empty([self.numGraphNodes, self.fNew], requires_grad=True)
            torch.nn.init.normal_(bias_val, -1, 1)
            bias_val = bias_val * self.denseInitializerScale
            self.bias = torch.nn.Parameter(bias_val)
        
        # input feature num
        self.F = self.fOld
    
    def forward(self, input_):
        batch_size = input_.shape[0]

        if self.denseConnect:
            h_new_final = torch.matmul(self.kernel, input_)
        else:
            h_old = torch.reshape(
                input_, [-1, self.numGraphNodes, self.F, 1]
            )
            h_old = h_old.expand(
                [-1, -1, -1, self.fNew]
            )
            
            h_new_final = h_old * self.kernel
            h_new_final = torch.sum(h_new_final, dim = 2)
            
            vectors = h_new_final.transpose(0, 1).reshape(self.numGraphNodes, -1)
            
            h_new_final = torch.sparse.mm(
                self.A, vectors
            ).reshape(self.numGraphNodes, batch_size, -1).transpose(1, 0)

        h_new_final = h_new_final + self.bias

        return h_new_final

class WootSpatialGCN(nn.Module):
    def __init__(
        self, 
        obj_reader = None, device = 'cpu',
        dense_initializer_scale = None, 
        feature_size1 = None, feature_size2 = None,
        use_batch_norm = None, fully_connected = None, ring_value = None,
        normalize = None, dense_inner_block = None,
        num_residual_blocks = None, input_size = None,
        output_size = None
    ):
        super(WootSpatialGCN, self).__init__()
    
        print('+++++ WootSpatialGCN: Start initalizing GCN Spatial GCN!')
        ####################################################################
        # the input params
        self.obj_reader = obj_reader
        self.device = device
        
        ####################################################################
        # the meta data
        self.A = None  # the adejacent matrix
        self.adj_st = None
        self.adj_ed = None
        self.adj_weight = None
        
        #################################################################### 
        if num_residual_blocks % 2 == 0:
            self.num_residual_blocks = num_residual_blocks
        else:
            print('Number of residual blocks has to be even!')
        
        # the meta data
        self.feature_size1              = int(feature_size1)
        self.feature_size2              = int(feature_size2)
        self.use_batch_norm             = bool(use_batch_norm)
        self.dense_initializer_scale    = float(dense_initializer_scale)
        self.num_graph_nodes            = self.obj_reader.numberOfVertices
        
        self.ring_value                 = int(ring_value) 
        self.normalize                  = bool(int(normalize))
        self.fully_connected            = fully_connected
        self.dense_inner_block          = int(dense_inner_block)
        self.input_size                 = int(input_size)
        self.output_size                = int(output_size)
        
        ####################################################################
        # compute the meta data
        self.init_adjacent_matrix()
        self.print_settings()

        ####################################################################
        # define the layers 
        self.gcnn_0 = GCNNOperator(
            fNew = self.feature_size2, fOld = self.input_size, denseInitializerScale=self.dense_initializer_scale,
            A=self.A, denseConnect=False, numGraphNodes=self.num_graph_nodes
        )

        ####################################################################
        # then set up the residue blocks 0
        for block_id in range(0, int(self.num_residual_blocks // 2)):
            # takes x and skip to generate x and x skip
            if self.use_batch_norm:
                cur_bn = torch.nn.BatchNorm1d(
                    num_features=self.num_graph_nodes * self.feature_size2
                )
                setattr(self, "bn_" + str(block_id) + '_0', cur_bn)
            
            cur_gcnn_0 = GCNNOperator(
                fNew = self.feature_size1, fOld = self.feature_size2, denseInitializerScale=self.dense_initializer_scale,
                A=self.A, denseConnect=False, numGraphNodes=self.num_graph_nodes
            )
            setattr(self, "gcnn_" + str(block_id) + '_0', cur_gcnn_0)

            if self.use_batch_norm:
                cur_bn = torch.nn.BatchNorm1d(
                    num_features=self.num_graph_nodes * self.feature_size1
                )
                setattr(self, "bn_" + str(block_id) + '_1', cur_bn)

            cur_gcnn_1 = GCNNOperator(
                fNew = self.feature_size2, fOld = self.feature_size1, denseInitializerScale=self.dense_initializer_scale,
                A=self.A, denseConnect=False, numGraphNodes=self.num_graph_nodes
            )
            setattr(self, "gcnn_" + str(block_id) + '_1', cur_gcnn_1)
        
        ####################################################################
        # then set up the dense inner blocks        
        if self.dense_inner_block:
            # takes x and skip to generate x and x skip
            if self.use_batch_norm:
                cur_bn = torch.nn.BatchNorm1d(
                    num_features=self.num_graph_nodes * self.feature_size2
                )
                setattr(self, "dense_0_bn", cur_bn)
            
            cur_dense_0 = GCNNOperator(
                fNew = self.feature_size2, fOld = self.feature_size2, 
                denseInitializerScale=self.dense_initializer_scale * 0.01,
                A=self.A, denseConnect=True, numGraphNodes=self.num_graph_nodes
            )
            
            setattr(self, "dense_0_gcnn", cur_dense_0)
        
        ####################################################################
        # then set up the residue blocks 1

        for block_id in range(int(self.num_residual_blocks // 2), self.num_residual_blocks, 1):
            # takes x and skip to generate x and x skip
            if self.use_batch_norm:
                cur_bn = torch.nn.BatchNorm1d(
                    num_features=self.num_graph_nodes * self.feature_size2,
                )
                setattr(self, "bn_" + str(block_id) + '_0', cur_bn)
            
            cur_gcnn_0 = GCNNOperator(
                fNew = self.feature_size1, fOld = self.feature_size2, denseInitializerScale=self.dense_initializer_scale,
                A=self.A, denseConnect=False, numGraphNodes=self.num_graph_nodes
            )
            setattr(self, "gcnn_" + str(block_id) + '_0', cur_gcnn_0)

            if self.use_batch_norm:
                cur_bn = torch.nn.BatchNorm1d(
                    num_features=self.num_graph_nodes * self.feature_size1
                )
                setattr(self, "bn_" + str(block_id) + '_1', cur_bn)

            cur_gcnn_1 = GCNNOperator(
                fNew = self.feature_size2, fOld = self.feature_size1, denseInitializerScale=self.dense_initializer_scale,
                A=self.A, denseConnect=False, numGraphNodes=self.num_graph_nodes
            )
            setattr(self, "gcnn_" + str(block_id) + '_1', cur_gcnn_1)

        #############################################################
        # setup final stage
        fin_gcnn = GCNNOperator(
            fNew = self.output_size, fOld = self.feature_size2, denseInitializerScale=self.dense_initializer_scale,
            A=self.A, denseConnect=False, numGraphNodes=self.num_graph_nodes
        )
        
        setattr(self, "final_gcnn", fin_gcnn)

        self.set_dense()

        print('+++++ WootSpatialGCN: End initalizing GCN Spatial GCN!')

    def set_dense(self):
        # hack for multi-processing
        self.A = self.A.to_dense()
        
        for att_name in dir(self):
            if "GCNNOperator" in str(type(getattr(self, att_name))):
                setattr(
                    getattr(self, att_name), 'A', getattr(getattr(self, att_name), 'A').to_dense()
                )
            
        return 

    def set_sparse(self):
        # another hack
        self.A = self.A.to_sparse()
        
        for att_name in dir(self):
            if "GCNNOperator" in str(type(getattr(self, att_name))):
                setattr(
                    getattr(self, att_name), 'A', getattr(getattr(self, att_name), 'A').to_sparse()
                )
        return 

    def print_settings(self):
        print(' ++ F1 size: ' + str(self.feature_size1))
        print(' ++ F2 size: ' + str(self.feature_size2))
        print(' ++ Use BatchNorm: ' + str(self.use_batch_norm))
        print(' ++ Init scale: ' + str(self.dense_initializer_scale))
        print(' ++ Number of Graph nodes: ' + str(self.num_graph_nodes))
        print(' ++ Ring value: ' + str(self.ring_value))
        print(' ++ Normalize: ' + str(self.normalize))
        print(' ++ Fully connected: ' + str(self.fully_connected))
        print(' ++ Number of residual blocks: ' + str(self.num_residual_blocks))
        print(' ++ Input size: ' + str(self.input_size))
        print(' ++ Output size: ' + str(self.output_size))

    def init_adjacent_matrix(self):
        print("+++++ WootSpatialGCN: start to compute adjacent!")
        # two more rings =D
        print('number of ring values', self.ring_value)
        if self.ring_value == -1:
            print("+++++ creating graph with the original adjacency!")
            print("+++++ however, not supported")
        else:
            print("+++++ creating graph with the new adjacency!")
            self.adj_st = []
            self.adj_ed = []
            self.adj_weight = []

            for i in range(self.num_graph_nodes):

                cur_ed, cur_weight = compute_neighbor_with_ring(
                    st_vert = i, 
                    neighbor_idx = self.obj_reader.verticesNeighborID,
                    distance_limit = self.ring_value,
                    normalize=self.normalize
                )
                cur_st = (np.ones_like(cur_ed) * i).astype(np.int32)
                
                self.adj_st.append(cur_st)
                self.adj_ed.append(cur_ed)
                self.adj_weight.append(cur_weight)
            
            self.adj_st = np.concatenate(self.adj_st, axis = 0)
            self.adj_ed = np.concatenate(self.adj_ed, axis = 0)
            self.adj_weight = np.concatenate(self.adj_weight, axis=0)
        
        self.adj_st = torch.LongTensor(self.adj_st).to(self.device)
        self.adj_ed = torch.LongTensor(self.adj_ed).to(self.device)
        #self.adj_zeros = torch.zeros_like(self.adj_st).to(self.device)
        self.adj_weight = torch.FloatTensor(self.adj_weight).to(self.device)

        self.A = torch.sparse_coo_tensor(
            indices = torch.stack([self.adj_st, self.adj_ed], dim = 0),
            values = self.adj_weight, size=(self.num_graph_nodes, self.num_graph_nodes), device=self.device,
            requires_grad=False
        )

        print("+++++ WootSpatialGCN: end to compute adjacent!")
        return 

    def forward(self, x):

        x = self.gcnn_0(x)
        xSkip = x

        #############################################################
        # residue block 1
        for block_id in range(0, int(self.num_residual_blocks // 2)):
            #x, xSkip = []
            if self.use_batch_norm:
                # now we need the batch norm
                cur_bn_0 = getattr(self, "bn_" + str(block_id) + '_0')
                # b, c, n
                x = x.reshape([-1, self.num_graph_nodes * self.feature_size2])
                x = cur_bn_0(x)
                x = x.reshape([-1, self.num_graph_nodes, self.feature_size2])
            
            x = torch.nn.functional.elu(x)

            cur_gcnn_0 = getattr(self, "gcnn_" + str(block_id) + '_0')
            x = cur_gcnn_0(x)
            
            if self.use_batch_norm:
                # now we need the batch norm
                cur_bn_1 = getattr(self, "bn_" + str(block_id) + '_1')
                # b, c, n
                x = x.reshape([-1, self.num_graph_nodes * self.feature_size1])
                x = cur_bn_1(x)
                x = x.reshape([-1, self.num_graph_nodes, self.feature_size1])
            
            cur_gcnn_1 = getattr(self, "gcnn_" + str(block_id) + '_1')
            x = cur_gcnn_1(x)
            
            x = xSkip + x
            xSkip = x
            
        #############################################################
        # dense block 1
        if self.dense_inner_block:
            # takes x and skip to generate x and x skip
            if self.use_batch_norm:
                cur_bn = getattr(self, "dense_0_bn")
                x = x.reshape([-1, self.num_graph_nodes * self.feature_size1])
                x = cur_bn(x)
                x = x.reshape([-1, self.num_graph_nodes, self.feature_size1])
            
            x = torch.nn.functional.elu(x)
            cur_dense = getattr(self, "dense_0_gcnn")
            x = cur_dense(x)
            x = xSkip + x
            xSkip = x

        #############################################################
        # residue block 2
        for block_id in range(int(self.num_residual_blocks // 2), self.num_residual_blocks, 1):
            if self.use_batch_norm:
                cur_bn_0 = getattr(self, "bn_" + str(block_id) + '_0')
                # b, c, n
                x = x.reshape([-1, self.num_graph_nodes * self.feature_size2])
                x = cur_bn_0(x)
                x = x.reshape([-1, self.num_graph_nodes, self.feature_size2])
            
            x = torch.nn.functional.elu(x)

            cur_gcnn_0 = getattr(self, "gcnn_" + str(block_id) + '_0')
            x = cur_gcnn_0(x)
            
            if self.use_batch_norm:
                cur_bn_1 = getattr(self, "bn_" + str(block_id) + '_1')
                # b, c, n
                x = x.reshape([-1, self.num_graph_nodes * self.feature_size1])
                x = cur_bn_1(x)
                x = x.reshape([-1, self.num_graph_nodes, self.feature_size1])
            
            cur_gcnn_1 = getattr(self, "gcnn_" + str(block_id) + '_1')
            x = cur_gcnn_1(x)
            x = xSkip + x
            xSkip = x
        
        #############################################################
        # final stage
        x = torch.nn.functional.elu(x)
        fin_gcnn = getattr(self, "final_gcnn")
        x = fin_gcnn(x)

        return x
