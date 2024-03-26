"""
@File: WootCharacter_ASH.py
@Author: Heming Zhu
@Email: hezhu@mpi-inf.mpg.de
@Date: 2023-09-25
@Desc: The pytorch learnable/not learnable embedded graph charactor (in Real-time Deep Dynamic Characters. Sigraph2021, Marc Habermann et.al ),
support for the charactor with/without hands. 
Could degrade to the skinning only version with the settings.
Modified for ASH to expose the quatarions.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from kornia.geometry.conversions import angle_axis_to_rotation_matrix, matrix4x4_to_Rt, rotation_matrix_to_quaternion, quaternion_to_rotation_matrix, Rt_to_matrix4x4
from kornia.geometry import convert_points_to_homogeneous
from kornia.geometry.quaternion import QuaternionCoeffOrder
import DeepCharacters_Pytorch.OBJReader as OBJReader

import trimesh
from DeepCharacters_Pytorch.WootSkeleton import WootSkeleton
from DeepCharacters_Pytorch.character_utils import dual_quad_to_trans_vec, batch_mat_to_angle, wrap_transformation_angle, compute_geodesic_distance, compute_trans_quad, batch_angle_to_mat

from einops import rearrange

class WootCharacterWithQuat(nn.Module):
    def __init__(
            self, 
            skeleton_dir = None, skinning_dir = None, template_mesh_dir=None, graph_dir = None, rest_pose_dir = None, 
            device='cpu', blending_type='dqs', deformation_type = 'embedded',
            compute_eg = False, compute_delta = True, compute_posed = False,
            hand_mask_dir = None, 
            use_sparse = True
        ):
        super(WootCharacterWithQuat, self).__init__()
        
        print('start to intialize character')
        self.skeleton_dir = skeleton_dir
        self.skinning_dir = skinning_dir
        self.template_mesh_dir = template_mesh_dir
        self.graph_dir = graph_dir
        self.rest_pose_dir = rest_pose_dir

        self.blending_type = blending_type
        self.deformation_type = deformation_type
        self.hand_mask_dir = hand_mask_dir

        self.device = device

        # if set False, then no pose deformation is applied (maybe it saves time)
        self.compute_eg = compute_eg
        self.compute_delta = compute_delta
        self.compute_pose = compute_posed
        
        # if wanna use it in dataloader (worker > 0) then should disable
        self.use_sparse = use_sparse
        ############################################################
        #                 character meta data
        ############################################################
        
        self.obj_reader = None
        self.vert_num = None
        self.temp_faces = None
        self.temp_verts = None
        self.temp_verts_homo = None
        self.temp_vert_colors = None
        self.temp_vert_normals = None
        self.temp_uv_coords = None

        self.rest_pose_global_transformation_mat = None
        self.rest_pose_local_translation_mat = None
        self.rest_pose_local_transformation_mat = None

        self.skinning_selection_mat = None
        self.skinning_id_to_skeleton_id = None
        self.skinning_joint_names = None
        self.skinning_joint_num = None

        self.skinning_weights_st = None
        self.skinning_weights_ed = None
        self.skinning_weights_iden = None
        self.skinning_weights_value = None
        self.skinning_weights_fst = None
        self.skinning_weights_dqs = None

        self.laplacian_temp_st = None
        self.laplacian_temp_ed = None
        self.laplacian_temp_weight = None
        self.sparse_laplacian = None
        self.laplacian_row_weight = None

        self.edge_temp_st = None
        self.edge_temp_ed = None

        # the mask regarding the hands
        self.hand_mask = None

        ############################################################
        #                 embedded graph meta data
        ############################################################

        self.graph_obj_reader = None
        self.graph_face = None
        self.graph_verts = None
        self.graph_verts_homo = None
        self.graph_verts_num = None
        # -> pointing the real mesh vertices
        self.graph_node_idx  = None
        
        self.highest_highres_vert_id = None
        self.lowest_highres_vert_id = None
    
        self.highest_lowres_vert_id = None
        self.lowest_lowres_vert_id = None

        self.link_temp_id = None
        self.link_node_id = None
        self.link_weight = None
        self.sparse_link_weight_matrix = None

        self.node_to_node_link_st = None
        self.node_to_node_link_ed = None
        self.first_skinning_joint_id = None
        
        ############################################################
        #                 load character file
        ############################################################

        # for the skeleton
        self.skeleton = WootSkeleton(
            self.skeleton_dir, device = self.device
        )
        
        self.load_template_mesh(self.template_mesh_dir)
        
        if self.hand_mask_dir is not None:
            self.load_hand_mask(self.hand_mask_dir)
        
        self.build_embedded_graph(self.graph_dir)

        self.load_skinning_file(self.skinning_dir)
        self.compute_rest_pose_translations()

        print('+++++ Fished initalizing Woot Character!')

    def load_hand_mask(self, hand_mask_dir):
        f = open(hand_mask_dir,'r').readlines()
        temp_mask_arr = np.array([(1 - min(int(f[i][0]),1) ) for i in range(len(f))])

        self.hand_mask = torch.FloatTensor(temp_mask_arr).to(self.device)
 
        return

    def load_template_mesh(self, mesh_dir):
        self.obj_reader = OBJReader.OBJReader(
            mesh_dir
        )
        
        self.temp_faces = torch.LongTensor(self.obj_reader.facesVertexId).reshape([-1, 3]).to(self.device)
        self.temp_vert_colors = torch.FloatTensor(self.obj_reader.vertexColors).reshape([-1, 3]).to(self.device)
        self.temp_uv_coords = torch.FloatTensor(self.obj_reader.textureCoordinates).reshape([-1,3,2]).to(self.device)
        self.temp_verts = torch.FloatTensor(self.obj_reader.vertexCoordinates).reshape([-1,3]).to(self.device)
        self.vert_num = self.obj_reader.numberOfVertices
        
        self.temp_verts_homo = torch.cat(
            [
                torch.FloatTensor(self.obj_reader.vertexCoordinates).reshape([-1,3]).to(self.device), 
                torch.ones([self.vert_num, 1]).float().to(self.device)
            ], axis=-1
        )

        # then compute the laplacian related link
        self.laplacian_temp_st = []
        self.laplacian_temp_ed = []
        self.laplacian_temp_weight = []

        self.edge_temp_st = []
        self.edge_temp_ed = []      

        for i in range(len(self.obj_reader.verticesNeighborID)):
            cur_st = []
            cur_ed = []
            cur_weight = []
            cur_arr = self.obj_reader.verticesNeighborID[i]
            
            cur_st.append(i)
            cur_ed.append(i)
            cur_weight.append(1.0)

            for j in range(len(cur_arr)):
                cur_st.append(i)
                cur_ed.append(cur_arr[j])

                if i < cur_arr[j]:
                    self.edge_temp_st.append(i)
                    self.edge_temp_ed.append(cur_arr[j])

                cur_weight.append(-1.0 / (1.0 * len(cur_arr)))

            self.laplacian_temp_ed.append(cur_ed)
            self.laplacian_temp_st.append(cur_st)
            self.laplacian_temp_weight.append(cur_weight)

        self.laplacian_temp_st = torch.LongTensor(np.concatenate(self.laplacian_temp_st, axis = 0)).to(self.device)
        self.laplacian_temp_ed = torch.LongTensor(np.concatenate(self.laplacian_temp_ed, axis = 0)).to(self.device)
        self.laplacian_temp_weight = torch.FloatTensor(np.concatenate(self.laplacian_temp_weight, axis = 0)).to(self.device)

        self.edge_temp_st = torch.LongTensor(self.edge_temp_st).to(self.device)
        self.edge_temp_ed = torch.LongTensor(self.edge_temp_ed).to(self.device)

        self.sparse_laplacian = torch.sparse_coo_tensor(
            indices= torch.stack([self.laplacian_temp_st, self.laplacian_temp_ed], dim = 0),
            values=self.laplacian_temp_weight, size=(self.vert_num, self.vert_num), device=self.device
        )
        
        self.sparse_laplacian = self.sparse_laplacian.coalesce()
        if not self.use_sparse:
            self.sparse_laplacian = self.sparse_laplacian.to_dense()
                
        return 

    def set_dense(self):
        print('+++++ Set Dense Character!')
        self.use_sparse = False
        self.sparse_laplacian = self.sparse_laplacian.to_dense()
        self.skinning_weights_dqs = self.skinning_weights_dqs.to_dense()
        self.skinning_weights_sprase = self.skinning_weights_sprase.to_dense()
        self.sparse_link_weight_matrix = self.sparse_link_weight_matrix.to_dense()
        return 
    
    def set_sparse(self):
        print('+++++ Set Sparse Character!')
        self.use_sparse = True
        self.sparse_laplacian = self.sparse_laplacian.to_sparse()
        self.skinning_weights_dqs = self.skinning_weights_dqs.to_sparse()
        self.skinning_weights_sprase = self.skinning_weights_sprase.to_sparse()
        self.sparse_link_weight_matrix = self.sparse_link_weight_matrix.to_sparse()
        return 

    def build_embedded_graph(self, graph_dir): 

        self.graph_obj_reader = OBJReader.OBJReader(graph_dir)
        
        self.graph_face = torch.LongTensor(self.graph_obj_reader.facesVertexId).reshape([-1, 3]).to(self.device)
        self.graph_verts = torch.FloatTensor(self.graph_obj_reader.vertexCoordinates).reshape([-1,3]).to(self.device)
        self.graph_verts_homo = torch.cat(
            [self.graph_verts, torch.ones([self.graph_obj_reader.numberOfVertices,1]).to(self.device)], dim = -1
        )
        self.graph_verts_num = self.graph_obj_reader.numberOfVertices
        
        t_graph_verts = np.array(self.graph_obj_reader.vertexCoordinates).reshape([-1, 3])
        t_obj_verts = np.array(self.obj_reader.vertexCoordinates).reshape([-1, 3])

        self.highest_lowres_vert_id = np.argmax(t_graph_verts[:,1])
        self.lowest_lowres_vert_id = np.argmin(t_graph_verts[:,1])
        
        self.highest_highres_vert_id = np.argmax(t_obj_verts[:,1])
        self.lowest_highres_vert_id = np.argmin(t_obj_verts[:,1])
        
        #####################################################################
                
        self.node_to_node_link_ed = []
        self.node_to_node_link_st = []

        for i in range(len(self.graph_obj_reader.verticesNeighborID)):
            for each_neighbor in self.graph_obj_reader.verticesNeighborID[i]:
                self.node_to_node_link_ed.append(each_neighbor)
                self.node_to_node_link_st.append(i)
            
        #####################################################################
        used_base_mesh_verts = np.array([False for i in range(self.vert_num)])
        
        embedded_node_idx = []
        embedded_node_neighbors = []
        
        embedded_node_radius = [] # geodistic distance 
        embedded_to_template_distance = [] # geodistic distance 

        for i in range(self.graph_verts_num):
            # find the cloest high-res mesh vertices for each graph node
            cur_graph_pt = t_graph_verts[i]
            graph_to_mesh_dist = np.sqrt(
                np.sum( (cur_graph_pt[None,...] - t_obj_verts) ** 2, axis= -1)
            ) + used_base_mesh_verts.astype(np.float32) * 1145141919.
            
            cloest_id = np.argmin(graph_to_mesh_dist)
            used_base_mesh_verts[cloest_id] = True
            
            # link between the gragh and the mesh
            embedded_node_idx.append(cloest_id)

            # add the neighbors on the template mesh to the graph
            embedded_node_neighbors.append(self.obj_reader.verticesNeighborID[cloest_id])

        for i in range(self.graph_verts_num):
            # compute the geodistics starting from the current embeded node
            cur_center = embedded_node_idx[i]
            
            cur_dist = compute_geodesic_distance(
                st_vert = cur_center, 
                neighbor_idx = self.obj_reader.verticesNeighborID,
                num_verts = self.obj_reader.numberOfVertices
            )
           
            cur_dist = np.array(cur_dist)
            
            cur_neighbor_id = np.array(embedded_node_neighbors[i])
            neighbor_dist = cur_dist[cur_neighbor_id]
    
            cur_node_radius = max(np.max(neighbor_dist) // 2, 3)
            
            embedded_to_template_distance.append(cur_dist)
            embedded_node_radius.append(cur_node_radius)
        
        embedded_node_radius = np.array(
            embedded_node_radius
        )
        embedded_to_template_distance = np.array(
            embedded_to_template_distance
        )
        
        temp_to_node_connect_node_id = [] 
        temp_to_node_connect_temp_id = []
        temp_to_node_connect_weight = []
        temp_to_node_connect_fst = []
        
        unconnected_vertices = 0

        max_connect = 0
        min_connect = 1145141919
        link_num = []

        # then fetch the connected vertices
        for i in range(self.vert_num):
            cur_vert_to_node_dist = embedded_to_template_distance[:,i] / (1.0 * embedded_node_radius)
            nearby_id = np.where(cur_vert_to_node_dist <= 1)[0]
                         
            if nearby_id.shape[0] < 1:
                unconnected_vertices +=1
                nearby_id = np.where(cur_vert_to_node_dist <= 2)[0]
                            
                nearby_dist = cur_vert_to_node_dist[nearby_id]
                nearby_weight = np.exp(-0.5 * nearby_dist * nearby_dist)
                
                temp_to_node_connect_node_id.append(nearby_id)
                temp_to_node_connect_temp_id.append(np.ones_like(nearby_id) * i)
                temp_to_node_connect_weight.append(nearby_weight)
                link_num.append(nearby_id.shape[0])
                
                max_connect = max(max_connect, nearby_dist.shape[0])
                min_connect = min(min_connect, nearby_dist.shape[0])
                
                temp_to_node_connect_fst.append(np.ones_like(nearby_id) * nearby_id[0])
                                
            else:
                  
                nearby_dist = cur_vert_to_node_dist[nearby_id]
                nearby_weight = np.exp(-0.5 * nearby_dist * nearby_dist)

                temp_to_node_connect_node_id.append(nearby_id)
                temp_to_node_connect_temp_id.append(np.ones_like(nearby_id) * i)
                temp_to_node_connect_weight.append(nearby_weight)
                link_num.append(nearby_id.shape[0])
                max_connect = max(max_connect, nearby_dist.shape[0])
                min_connect = min(min_connect, nearby_dist.shape[0])
                
                temp_to_node_connect_fst.append(np.ones_like(nearby_id) * nearby_id[0])
        
        
        # then normalize the weights 
        for i in range(self.vert_num):
            temp_to_node_connect_weight[i] = temp_to_node_connect_weight[i] / np.sum(temp_to_node_connect_weight[i])
        
        #####################################################################
        
        self.graph_node_idx = torch.LongTensor(embedded_node_idx).to(self.device)
        # set up the link
        temp_to_node_connect_node_id = np.concatenate(temp_to_node_connect_node_id, axis = 0)
        temp_to_node_connect_temp_id = np.concatenate(temp_to_node_connect_temp_id, axis = 0)
        temp_to_node_connect_weight = np.concatenate(temp_to_node_connect_weight, axis = 0)
        temp_to_node_connect_fst = np.concatenate(temp_to_node_connect_fst, axis = 0)
        
        # the node -> vert link
        self.link_temp_id = torch.LongTensor(temp_to_node_connect_temp_id).to(self.device)
        self.link_node_id = torch.LongTensor(temp_to_node_connect_node_id).to(self.device)
        self.link_weight = torch.FloatTensor(temp_to_node_connect_weight).to(self.device)
        self.link_fst = torch.LongTensor(temp_to_node_connect_fst).to(self.device)
                                
        iden_idx = torch.linspace(
            start=0, end = self.link_weight.shape[0] - 1, steps=self.link_weight.shape[0]
        ).long().to(self.device)

        # then we can also create a sparse tensor for the weighting
        self.sparse_link_weight_matrix = torch.sparse_coo_tensor(
            indices = torch.stack([self.link_temp_id, iden_idx], dim = 0),
            values = self.link_weight, size=(self.obj_reader.numberOfVertices, self.link_temp_id.shape[0]), device=self.device
        )
        self.sparse_link_weight_matrix = self.sparse_link_weight_matrix.coalesce()
        
        if not self.use_sparse:
            self.sparse_link_weight_matrix = self.sparse_link_weight_matrix.to_dense()

        # the node -> node link
        self.node_to_node_link_ed = torch.LongTensor(self.node_to_node_link_ed).to(self.device)
        self.node_to_node_link_st = torch.LongTensor(self.node_to_node_link_st).to(self.device)
        
        return 

    def load_skinning_file(self, skinning_dir):
        skinning_data_block = open(skinning_dir).readlines()

        skinning_name_line = skinning_data_block[2].rstrip().split()

        skeleton_joint_names= [
            t.split('_')[0] for t in self.skeleton.joint_name_cache
        ]

        last_id_list = []
        last_names_list = []

        # the joints to pick
        for i in range(len(skinning_name_line)):
            found_id = False
            for j in range(len(skeleton_joint_names) - 1, 0, -1):
                if skeleton_joint_names[j] == skinning_name_line[i]:
                    found_id = True
                    last_id_list.append(j)
                    last_names_list.append(self.skeleton.joint_name_cache[j])
                    break
            if not found_id:
                print('missing joints', skinning_name_line[i])
                sys.exit(0)
        
        self.skinning_id_to_skeleton_id = torch.LongTensor(last_id_list).to(self.device)
        self.skinning_joint_names = last_names_list
        self.skinning_joint_num = self.skinning_id_to_skeleton_id.shape[0]
        
        self.skinning_weights_st = []
        self.skinning_weights_ed = []
        self.skinning_weights_iden = []
        self.skinning_weights_fst = []
        self.skinning_weights_value = []

        # start to load skinning
        skinning_weight_file_offset = 4

        temp_link_id_num = 0

        for i in range(self.vert_num):
            cur_line = skinning_data_block[i + skinning_weight_file_offset].split()
            cur_joint_num = (len(cur_line) - 1) // 2
            cur_first_idx = -1
            for j in range(cur_joint_num):
                cur_id, cur_weight = int(cur_line[j * 2  + 1]), float(cur_line[j * 2  + 2])

                if j == 0:                    
                    cur_first_idx = cur_id

                self.skinning_weights_st.append(i)
                self.skinning_weights_ed.append(cur_id)
                self.skinning_weights_iden.append(temp_link_id_num)
                self.skinning_weights_value.append(cur_weight)
                self.skinning_weights_fst.append(cur_first_idx)
                
                temp_link_id_num += 1
                
        self.skinning_weights_st = torch.LongTensor(self.skinning_weights_st).to(self.device)
        self.skinning_weights_ed = torch.LongTensor(self.skinning_weights_ed).to(self.device)
        self.skinning_weights_iden = torch.LongTensor(self.skinning_weights_iden).to(self.device)
        
        self.skinning_weights_value = torch.FloatTensor(self.skinning_weights_value).to(self.device)
        self.skinning_weights_fst = torch.LongTensor(self.skinning_weights_fst).to(self.device)
        
        
        self.skinning_weights_dqs = torch.sparse_coo_tensor(
            indices= torch.stack([self.skinning_weights_st, self.skinning_weights_iden], dim = 0),
            values=self.skinning_weights_value, size=(self.vert_num, temp_link_id_num), device=self.device
        )
        self.skinning_weights_dqs = self.skinning_weights_dqs.coalesce()
        if not self.use_sparse:
            self.skinning_weights_dqs = self.skinning_weights_dqs.to_dense()
        
        self.skinning_weights_sprase = torch.sparse_coo_tensor(
            indices= torch.stack([self.skinning_weights_st, self.skinning_weights_ed], dim = 0),
            values=self.skinning_weights_value, size=(self.vert_num, self.skinning_joint_num), device=self.device
        )
        self.skinning_weights_sprase = self.skinning_weights_sprase.coalesce()
        if not self.use_sparse:
            self.skinning_weights_sprase = self.skinning_weights_sprase.to_dense()

        return 

    def compute_rest_pose_translations(self):
        # load rest_pose dof
        rest_pose_dof_data_block = open(self.rest_pose_dir,'r').readlines()
        rest_pose_dof = rest_pose_dof_data_block[1].split()
        rest_pose_dof = torch.FloatTensor([float(t) for t in rest_pose_dof[1:]]).unsqueeze(0).to(self.device)

        ret_global_transform , ret_local_trasnlation, _ = self.skeleton.forward(
            rest_pose_dof
        )

        self.rest_pose_global_transformation_mat = ret_global_transform[:,self.skinning_id_to_skeleton_id,:,:]
        self.rest_pose_global_transformation_mat_inv = torch.linalg.inv(self.rest_pose_global_transformation_mat)

        self.rest_pose_local_translation_mat = ret_local_trasnlation[:,self.skinning_id_to_skeleton_id,:,:]

        return 

    def dqs_blending(self, translation_mat):
        
        batch_size = translation_mat.shape[0]
        joint_num = translation_mat.shape[1]
               
        translation_vec = translation_mat[...,:3, 3]
        rotation_mat = translation_mat[...,:3,:3]
        
        rot_quad = rotation_matrix_to_quaternion(
            rotation_mat.view([-1, 3, 3]).contiguous(),
            order = QuaternionCoeffOrder.WXYZ
        ).view([batch_size, joint_num, -1])

        normalized_rot_quad = torch.nn.functional.normalize(rot_quad, dim = -1)
        
        translation_quad = compute_trans_quad(
            q = rot_quad,
            t = translation_vec
        )

        # get the skinning joints quaterians
        selected_rot_quad = normalized_rot_quad[:,self.skinning_weights_ed,:]
        selected_trans_quad = translation_quad[:,self.skinning_weights_ed,:]
        selected_first_rot_quad = normalized_rot_quad[:,self.skinning_weights_fst,:]

        sign_to_first = (torch.sum(
            selected_first_rot_quad * selected_rot_quad, dim = -1
        ) > 0).float()

        fin_sign = sign_to_first * 2 - 1

        link_translation = selected_trans_quad * fin_sign[...,None]
        link_rotation = selected_rot_quad * fin_sign[...,None]

        link_translation = rearrange(
            link_translation, 'b l c -> l (b c)'
        )

        link_rotation = rearrange(
            link_rotation, 'b l c -> l (b c)'
        )

        if not self.use_sparse:
            weighted_translation = torch.mm(
                self.skinning_weights_dqs, link_translation
            ).reshape([self.vert_num, batch_size, 4])

            weighted_rotation= torch.mm(
                self.skinning_weights_dqs, link_rotation
            ).reshape([self.vert_num, batch_size, 4])
        else:  
            weighted_translation = torch.sparse.mm(
                self.skinning_weights_dqs, link_translation
            ).reshape([self.vert_num, batch_size, 4])
            
            weighted_rotation= torch.sparse.mm(
                self.skinning_weights_dqs, link_rotation
            ).reshape([self.vert_num, batch_size, 4])         

        weighted_translation = rearrange(
            weighted_translation, 'v b c -> b v c'
        )

        weighted_rotation = rearrange(
            weighted_rotation, 'v b c -> b v c'
        )

        raw_scale = torch.norm(
            weighted_rotation, p=2, dim = -1
        )

        scale_mask = (raw_scale < 1e-9).float()

        fin_scale = 1. / (raw_scale + scale_mask)

        weighted_rotation = weighted_rotation * fin_scale[...,None]
        weighted_translation = weighted_translation * fin_scale[...,None]

        fin_rot_mat = quaternion_to_rotation_matrix(
            weighted_rotation.reshape([-1,4]), QuaternionCoeffOrder.WXYZ
        ).reshape([-1, 3, 3])

        fin_trans_vec = dual_quad_to_trans_vec(
            weighted_rotation, weighted_translation
        ).reshape([-1, 3])

        fin_transform_mat = Rt_to_matrix4x4(
            fin_rot_mat, fin_trans_vec[...,None]
        ).reshape([batch_size, self.vert_num, 4, 4])

        return fin_transform_mat

    def lbs_blending(self, transformation_mat):

        batch_size = transformation_mat.shape[0]

        temp_transformation_mat = transformation_mat.reshape([batch_size, self.skinning_joint_num, 16])
        temp_transformation_mat = temp_transformation_mat.transpose(0, 1).reshape(self.skinning_joint_num, -1)
        
        if self.use_sparse:
            weighted_transformation = torch.sparse.mm(
                self.skinning_weights_sprase, temp_transformation_mat
            ).reshape([self.vert_num, batch_size, 16])
        else:
            weighted_transformation = torch.mm(
                self.skinning_weights_sprase, temp_transformation_mat
            ).reshape([self.vert_num, batch_size, 16])            

        weighted_transformation = weighted_transformation.transpose(0, 1)
        weighted_transformation = weighted_transformation.reshape([batch_size, self.vert_num , 4, 4])

        return weighted_transformation

    def compute_posed_template_embedded_graph(self, dof = None, cached_global_transform = None):

        if not(cached_global_transform is None):
            cur_global_tranform = cached_global_transform
            batch_size = cur_global_tranform.shape[0]
        else:
            cur_global_tranform, _, _ = self.skeleton.forward(
                dof
            )

            batch_size = dof.shape[0]
        
        picked_global_transform = cur_global_tranform[:,self.skinning_id_to_skeleton_id,:,:]

        org_translation_mat = picked_global_transform
        
        org_translation_mat_0 = (
            picked_global_transform @ torch.linalg.inv(self.rest_pose_global_transformation_mat)
        )
        
        if self.blending_type == 'lbs':
            blended_mat_0 = self.lbs_blending(
                org_translation_mat
            )
            blended_mat_1 = self.lbs_blending(
                org_translation_mat_0
            )
        elif self.blending_type == 'dqs':
            blended_mat_0 = self.dqs_blending(
                org_translation_mat
            ) 
            blended_mat_1 = self.dqs_blending(
                org_translation_mat_0
            ) 
        
        # embedded graph
        ##########################################################################
        picked_blended_mat_0 = blended_mat_0[:,self.graph_node_idx,:,:]

        pickedR_mat, pickedT = matrix4x4_to_Rt(
            picked_blended_mat_0.reshape(-1, 4, 4)
        )

        pickedEularR = batch_mat_to_angle(
            pickedR_mat
        )

        pickedEularR = pickedEularR.reshape([batch_size,self.graph_verts_num,3])
        pickedT = pickedT[...,0].reshape([batch_size,self.graph_verts_num,3])

        ##########################################################################

        batch_size = blended_mat_1.shape[0]
        
        picked_blended_mat_1 = (blended_mat_1[:,self.graph_node_idx,:,:])[:,self.link_node_id, :, :]
        
        picked_temp_verts_nr = (
            picked_blended_mat_1 @ self.temp_verts_homo[..., self.link_temp_id, :][...,None]
        )[...,0]
        
        vectors = picked_temp_verts_nr.transpose(0, 1).reshape(self.link_node_id.shape[0], -1)

        if self.use_sparse:
            ret_posed_template = torch.sparse.mm(
                self.sparse_link_weight_matrix, vectors
            )
        else:
            ret_posed_template = torch.mm(
                self.sparse_link_weight_matrix, vectors
            )            

        ret_posed_template = ret_posed_template.reshape([-1, batch_size, 4]).transpose(0, 1)
        
        ret_posed_template = ret_posed_template[...,:3] / ret_posed_template[...,3:]

        return ret_posed_template, pickedEularR, pickedT

    def compute_embedded_graph_deformation_test(self, blended_mat = None, deltaR = None, deltaT = None, perVertex_displacement=None):
        
        batch_size = blended_mat.shape[0]
        org_template = self.temp_verts
        
        deltaR_mat = batch_angle_to_mat(deltaR)

        picked_temp_verts = org_template[self.link_temp_id,:]
        picked_node_verts = org_template[
            self.graph_node_idx[self.link_node_id],:
        ]

        picked_deltaR_mat = deltaR_mat[:,self.link_node_id,:,:]
        picked_deltaT = deltaT[:,self.link_node_id]

        v_nr = (picked_deltaR_mat @ (picked_temp_verts - picked_node_verts)[..., None])[...,0] + picked_node_verts + picked_deltaT
            
        vectors = v_nr.transpose(0, 1).reshape(self.link_node_id.shape[0], -1)

        eg_canoical = torch.mm(self.sparse_link_weight_matrix, vectors)

        eg_canoical = eg_canoical.reshape([self.vert_num, -1, 3]).transpose(0, 1)
        eg_canoical = convert_points_to_homogeneous(eg_canoical)

        delta_canoical = eg_canoical
        delta_canoical[...,:3] += perVertex_displacement
        
        if self.deformation_type == 'embedded':
              
            picked_blended_mat = (blended_mat[:,self.graph_node_idx,:,:])[:,self.link_node_id, :, :]

            link_num =  picked_blended_mat.shape[1]
            translation_vec = picked_blended_mat[...,:3, 3]
            rotation_mat = picked_blended_mat[...,:3,:3]

            rot_quad = rotation_matrix_to_quaternion(
                rotation_mat.view([-1, 3, 3]).contiguous(),
                order = QuaternionCoeffOrder.WXYZ
            ).view([batch_size, link_num, -1])

            normalized_rot_quad = torch.nn.functional.normalize(rot_quad, dim = -1)
            selected_first_rot_quad = normalized_rot_quad[:,self.link_fst,:]
            sign_to_first = (torch.sum(
                selected_first_rot_quad * normalized_rot_quad, dim = -1
            ) > 0).float()

            fin_sign = sign_to_first * 2 - 1
            
            translation_quad = compute_trans_quad(
                q = rot_quad,
                t = translation_vec
            )
            
            
            translation_quad = translation_quad * fin_sign[...,None]
            rot_quad = rot_quad * fin_sign[...,None]

            translation_quad = rearrange(
                translation_quad, 'b l c -> l (b c)'
            )

            rot_quad = rearrange(
                rot_quad, 'b l c -> l (b c)'
            )

            weighted_translation = torch.mm(
                self.sparse_link_weight_matrix, translation_quad
            ).reshape([self.vert_num, batch_size, 4])
            
            weighted_rotation = torch.mm(
                self.sparse_link_weight_matrix, rot_quad
            ).reshape([self.vert_num, batch_size, 4])

            weighted_translation = rearrange(
                weighted_translation, 'v b c -> b v c'
            )

            weighted_rotation = rearrange(
                weighted_rotation, 'v b c -> b v c'
            )

            raw_scale = torch.norm(
                weighted_rotation, p=2, dim = -1
            )

            scale_mask = (raw_scale < 1e-9).float()

            fin_scale = 1. / (raw_scale + scale_mask)

            weighted_rotation = weighted_rotation * fin_scale[...,None]
            weighted_translation = weighted_translation * fin_scale[...,None]

            fin_rot_mat = quaternion_to_rotation_matrix(
                weighted_rotation.reshape([-1,4]), QuaternionCoeffOrder.WXYZ
            ).reshape([-1, 3, 3])

            fin_trans_vec = dual_quad_to_trans_vec(
                weighted_rotation, weighted_translation
            ).reshape([-1, 3])

            fin_transform_mat = Rt_to_matrix4x4(
                fin_rot_mat, fin_trans_vec[...,None]
            ).reshape([batch_size, self.vert_num, 4, 4])
            
            ret_posed_delta = (fin_transform_mat @ delta_canoical[...,None])[...,0]
            ret_posed_delta = ret_posed_delta[...,:3] / ret_posed_delta[...,3:]
                
        else:
            print('deformation type', self.deformation_type, ' not supported')
                            
        return ret_posed_delta, delta_canoical, weighted_translation, weighted_rotation

    def forward_test(self, dof = None, delta_R = None, delta_T = None, per_vertex_T = None, cached_global_transform = None):
        """
            Parameters:
            dof - skelton dof
            delta_R - embedded graph rotation
            delta_T - embedded graph translation 
            per_vertex_T - per-vertex transformation on the mesh template
            
            Returns:
            ret_posed_delta - posed template with embedded and per-vertex deformation   
            delta_canoical - canonical template with embedded and per-vertex deformation   
            fin_translation_quad - translation quaterion for each mesh vertex 
            fin_rotation_quad - rotation quaterion for each mesh vertex 
            cur_global_tranform - skeleton joint positions
        """
 
        if not (cached_global_transform is None):
            cur_global_tranform = cached_global_transform
        else:
            cur_global_tranform, _, _ = self.skeleton.forward(
                dof
            )

        picked_global_transform = cur_global_tranform[:,self.skinning_id_to_skeleton_id,:,:]

        org_translation_mat = (
            picked_global_transform @ self.rest_pose_global_transformation_mat_inv
        )
        
        if self.blending_type == 'lbs':
            blended_mat = self.lbs_blending(
                org_translation_mat
            )
        elif self.blending_type == 'dqs':
            blended_mat = self.dqs_blending(
                org_translation_mat
            )
        
        ret_posed_delta, delta_canoincal, fin_translation_quad, fin_rotation_quad = self.compute_embedded_graph_deformation_test(
            blended_mat = blended_mat, 
            deltaR = delta_R, deltaT = delta_T, 
            perVertex_displacement = per_vertex_T
        )

        return ret_posed_delta, delta_canoincal, fin_translation_quad, fin_rotation_quad, cur_global_tranform[:,:, :3, -1]
