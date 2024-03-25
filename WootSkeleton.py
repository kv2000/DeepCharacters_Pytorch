"""
@File: WootSkeleton.py
@Author: Heming Zhu
@Email: hezhu@mpi-inf.mpg.de
@Date: 2023-09-25
@Desc: The (minimal usalbe) pytorch skeleton implementation for the character models in DynaCap(later) dataset, which consumes the dofs to generate the joint translations/transformations. 
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from einops import rearrange
import kornia
from pytorch3d.transforms import axis_angle_to_matrix
import woot_cuda_skeleton_fin as woot_cuda_skeleton

class WootSkeleton(nn.Module):
    def __init__(
            self, skeleton_dir = None, device='cpu'
        ):
        super(WootSkeleton, self).__init__()
        
        ############################################################
        #           start  the skeleton related infos
        ############################################################
        self.skeleton_version = None
        self.skeleton_scale = 1.0
        self.rest_pose = None

        self.min_limit = -1e9
        self.max_limit = 1e9
        self.device = device

        ############################################################
        #                   joints defination
        ############################################################
        self.m_joints = []
        self.parent_name_cache = []
        self.joint_name_cache = []
        self.parent_id_cache = []
        self.m_root = -1
        # the base joints, also we call it the real joints
        self.base_joint_id = []
        self.skin_joint_id = []

        self.number_of_joints = -1
       
        ############################################################
        #                   markers defination
        ############################################################
        self.m_markers = []
        self.number_of_markers = -1

        ############################################################
        #                scaling joints defination
        ############################################################
        self.number_of_scaling = -1
    
        ############################################################
        #                    dof defination
        ############################################################
        self.number_of_dofs = -1
        self.m_dofs = []

        ############################################################
        #                   rest pose intialization
        ############################################################
        self.rest_pose_full_joints = []
        self.rest_pose_skinned_joints = []

        ############################################################
        #                    stored output for c++
        ############################################################

        self.joint_axis_arr = []
        self.joint_scale_arr = []
        self.joint_offset_arr = []
        self.joint_type_arr = []
        self.joint_is_revolute = []
        self.joint_is_prismatic = []
        self.local_joint_translation = []

        ############################################################
        #                   starts intialization
        ############################################################
        self.init_skeleton(skeleton_dir)
 
    def build_skeleton_ver_1_0(self, data_block):
        self.skeleton_version = data_block[0].rstrip().split(' ')[-1][1:]
        self.m_joints = []
        self.parent_name_cache = []
        self.parent_id_cache = []
        self.parent_id_layered = []
        self.current_id_layered = []

        self.joint_name_cache = []
        self.m_root = -1
        
        self.base_joint_id = []

        self.joint_axis_arr = []
        self.joint_scale_arr = []
        self.joint_offset_arr = []
        self.joint_type_arr = []

        self.local_joint_translation = []
        temp_local_joint_translation = []

        ##########################################################################
        #                               load joints
        ##########################################################################
        self.number_of_joints = int(data_block[1].rstrip().split(' ')[-1])
        l_base_joints = 2
        
        for i in range(l_base_joints, l_base_joints + self.number_of_joints):
            cur_line = data_block[i].rstrip().split()
            joint_id = i - l_base_joints       
            joint_name, type_name, parent_name =  cur_line[0], cur_line[1], cur_line[2] 
            ox, oy, oz = float(cur_line[3]), float(cur_line[4]), float(cur_line[5]) 
            ax, ay, az = float(cur_line[6]), float(cur_line[7]), float(cur_line[8]) 
            sc = float(cur_line[9]) 

            if (type_name == "revolute"):          
                current_joint = {
                    'offset': torch.FloatTensor([ox, oy, oz]).to(self.device),
                    'axis': F.normalize(torch.FloatTensor([ax, ay, az]),dim=-1).to(self.device),
                    'scale': sc,
                    'id': joint_id,
                    'type_name': type_name,
                    'joint_name': joint_name,
                    'children_joints': [],
                    'base_joint_id': -1
                }

                self.joint_offset_arr.append([ox, oy, oz])
                self.joint_axis_arr.append(F.normalize(torch.FloatTensor([ax, ay, az]),dim=-1).numpy().tolist())
                self.joint_scale_arr.append(sc)
                self.joint_type_arr.append(type_name)
                self.joint_is_revolute.append(True)
                self.joint_is_prismatic.append(False)

            elif (type_name == "prismatic"):
                current_joint = {
                    'offset': torch.FloatTensor([ox, oy, oz]).to(self.device),
                    'axis': torch.FloatTensor([ax, ay, az]).to(self.device),
                    'scale': sc,
                    'id': joint_id,
                    'type_name': type_name,
                    'joint_name': joint_name,
                    'children_joints': [],
                    'base_joint_id': -1
                }
                
                self.joint_offset_arr.append([ox, oy, oz])
                self.joint_axis_arr.append(F.normalize(torch.FloatTensor([ax, ay, az]),dim=-1).numpy().tolist())
                self.joint_scale_arr.append(sc)
                self.joint_type_arr.append(type_name)
                self.joint_is_revolute.append(False)
                self.joint_is_prismatic.append(True)

            else:
                print('unkown joint type')
                sys.exit(0)
            

            temp_local_joint_translation.append(
                [ox * sc, oy * sc, oz * sc]
            )

            self.m_joints.append(current_joint)
            self.parent_name_cache.append(parent_name)
            self.joint_name_cache.append(joint_name)
               
        # find parents
        for i in range(self.number_of_joints):
            if self.parent_name_cache[i] in self.joint_name_cache:
                cur_parent_joint_id = self.joint_name_cache.index(self.parent_name_cache[i])
                self.parent_id_cache.append(cur_parent_joint_id)
            else:
                self.m_root = i
                self.parent_id_cache.append(-1)
        
        # find base points
        for i in range(self.number_of_joints):
            p = i
            while (not self.parent_id_cache[p] == -1) and (not(torch.norm(self.m_joints[p]['offset'], dim=-1) > 0)):
                p = self.parent_id_cache[p]
            
            if not p in self.base_joint_id:
                self.base_joint_id.append(p)

            self.m_joints[i]['base_joint_id'] = p
            
            if (not (p == i)) and (not (i in self.m_joints[p]['children_joints'])):
                self.m_joints[p]['children_joints'].append(i)
        
        self.current_id_layered, self.parent_id_layered = self.compute_layered_parent_id(self.parent_id_cache)
        
        for i in range(len(self.current_id_layered)):
            self.current_id_layered[i] = torch.LongTensor(self.current_id_layered[i]).to(self.device)
            self.parent_id_layered[i] = torch.LongTensor(self.parent_id_layered[i]).to(self.device)
        
        ##########################################################################
        #                               load markers
        ##########################################################################      
        # load markers
        # marker discarded
        l_base_markers = l_base_joints + self.number_of_joints + 1
        self.number_of_markers = int(data_block[l_base_markers - 1 ].rstrip().split(' ')[-1])
                
        ##########################################################################
        #                               load scaling joints
        ##########################################################################      
        # load scaling joints
        l_base_scaling_joints = l_base_markers + self.number_of_markers
        self.number_of_scaling = int(data_block[l_base_scaling_joints].rstrip().split(' ')[-1])

        ##########################################################################
        #                               load dofs
        ##########################################################################      
        # load dof infos 
        l_base_dof = l_base_scaling_joints + self.number_of_scaling + 1
        self.number_of_dofs = int(data_block[l_base_dof].rstrip().split(' ')[-1])

        # -> we make a sparse dof
        # dof st -> is the joints, dof ed -> is the dofs 
        self.dof_st = []
        self.dof_ed = []
        self.dof_weights = []
        
        tmp_base = l_base_dof + 1
        for i in range(self.number_of_dofs):
            name_line = data_block[tmp_base].rstrip().split()
            dof_name, dof_num = name_line[0], int(name_line[1])

            limit_line = data_block[tmp_base + 1].rstrip().split()
            
            cur_limit_min, cur_limit_max = self.min_limit, self.max_limit

            if limit_line[0] == 'limits':
                cur_limit_min, cur_limit_max = float(limit_line[1]), float(limit_line[2])
            
            influence_joint_list = []
            influence_weight_list = []
            
            for j in range(dof_num):
                dof_line = data_block[tmp_base + 2 + j].rstrip().split()
                influence_joint_name, influence_joint_weight = dof_line[0], float(dof_line[1])
                
                joint_idx = self.joint_name_cache.index(influence_joint_name)
                influence_joint_list.append(joint_idx)
                influence_weight_list.append(influence_joint_weight)

                self.dof_st.append(joint_idx)
                self.dof_ed.append(i)
                self.dof_weights.append(influence_joint_weight)

            self.m_dofs.append({
                'limit_min' : cur_limit_min,
                'limit_max' : cur_limit_max,
                'joint_idx': torch.LongTensor(influence_joint_list).to(self.device),
                'weights':  torch.FloatTensor(influence_weight_list).to(self.device),
                'dof_name' : dof_name, 
                'dof_num' : dof_num
            })

            current_block_size = 2 + dof_num
            tmp_base += current_block_size
        
        self.dof_st = torch.LongTensor(self.dof_st).to(self.device)
        self.dof_ed = torch.LongTensor(self.dof_ed).to(self.device)
        self.dof_weights = torch.FloatTensor(self.dof_weights).to(self.device)

        self.joint_dof_mat = torch.sparse_coo_tensor(
            indices= torch.stack([self.dof_st, self.dof_ed], dim = 0),
            values=self.dof_weights, size=(self.number_of_joints, self.number_of_dofs), device=self.device
        )

        self.joint_dof_mat = self.joint_dof_mat.to_dense()

        self.joint_axis_arr = torch.FloatTensor(self.joint_axis_arr)[None,...].to(self.device)
        self.parent_id_cache_arr = torch.LongTensor(self.parent_id_cache).to(self.device)
        self.joint_is_revolute = torch.BoolTensor(self.joint_is_revolute)[None,...].float().to(self.device)
        self.joint_is_prismatic = torch.BoolTensor(self.joint_is_prismatic)[None,...].float().to(self.device)

        # the translation
        self.local_joint_translation = np.expand_dims(np.eye(4), 0).repeat(self.number_of_joints, axis=0)
        self.local_joint_translation[:,:3,3] = np.array(temp_local_joint_translation)
        self.local_joint_translation = torch.FloatTensor(self.local_joint_translation)[None,...].to(self.device)

        return 
    
    def compute_layered_parent_id(self, org_parent_id):
        ret_cur_idx = []
        ret_fa_idx = []

        fa_num = 0
        fa_stat = np.array(org_parent_id)
        occ_stat = np.ones(fa_stat.shape[0]).astype(np.bool)
        org_parent_id = np.array(org_parent_id)

        while True:
            cur_enqueue = np.where(np.logical_and(fa_stat == -1, occ_stat))[0] 
            temp_cur_idx = []
            temp_fa_idx = []

            for i in range(cur_enqueue.shape[0]):
                temp_cur_idx.append(cur_enqueue[i])
                
                if org_parent_id[cur_enqueue[i]] == -1:
                    temp_fa_idx.append(cur_enqueue[i])
                else:
                    temp_fa_idx.append(org_parent_id[cur_enqueue[i]])
                
                occ_stat[cur_enqueue[i]] = 0 
                new_enqueue = np.where(org_parent_id == cur_enqueue[i])[0]
                fa_stat[new_enqueue] = -1

            fa_num += len(temp_fa_idx)

            ret_cur_idx.append(temp_cur_idx)
            ret_fa_idx.append(temp_fa_idx)

            if fa_num == len(org_parent_id):
                break
        
        return ret_cur_idx, ret_fa_idx
    
    def init_skeleton(self, skeleton_file_name):
        if not os.path.isfile(skeleton_file_name):
            print('Skeleton File Not Found')
            sys.exit(0)

        # first decide the type of the skeleton
        data_block = open(skeleton_file_name, 'r').readlines()
        temp_skeleton_version = data_block[0].rstrip().split(' ')[-1][1:]
        
        if temp_skeleton_version == '1.0':
            self.build_skeleton_ver_1_0(data_block)
        else:
            print('not supported skeleton version')
    
        return 
    
    def forward(self, dof):
        
        batch_num = dof.shape[0]
     
        ret_local_joint_translation = self.local_joint_translation.expand([batch_num, -1, -1, -1])
        
        dof_vec = rearrange(dof, 'b c -> c b')
        weighted_dof = torch.mm(self.joint_dof_mat, dof_vec)
        ret_joint_param = rearrange(weighted_dof, 'b c -> c b')

        ret_joint_prism = ret_joint_param * self.joint_is_prismatic
        ret_joint_revolute = ret_joint_param * self.joint_is_revolute

        prism_vec = ret_joint_prism[...,None] * self.joint_axis_arr
        
        revolute_vec = ret_joint_revolute[...,None] * self.joint_axis_arr
        revolute_vec = revolute_vec.reshape([-1, 3])

        revolute_mat = axis_angle_to_matrix(revolute_vec)

        prism_vec = prism_vec.reshape([-1, 3])

        ret_local_joint_transformation = kornia.geometry.conversions.Rt_to_matrix4x4(
            revolute_mat, prism_vec[...,None]
        )

        ret_local_joint_transformation = ret_local_joint_transformation.reshape([batch_num, self.number_of_joints, 4, 4])

        ret_local_joint_transformation_fin = ret_local_joint_translation @ ret_local_joint_transformation

        ret_joint_transformation = woot_cuda_skeleton.skeleton_fw(
            self.number_of_joints,
            self.current_id_layered,
            self.parent_id_layered,
            ret_local_joint_transformation_fin
        )
    
        return ret_joint_transformation, ret_local_joint_translation, ret_local_joint_transformation