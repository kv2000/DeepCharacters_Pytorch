"""
@File: 2_test_deformable_character.py
@Author: Heming Zhu
@Email: hezhu@mpi-inf.mpg.de
@Date: 2024-03-25
@Desc: Some test scripts for the deformable character.
"""

import sys
sys.path.append("../")
sys.path.append("../../")

import os
from pyhocon import ConfigFactory
from argparse import ArgumentParser, Namespace

import torch
import numpy as np

from WootSkeleton import WootSkeleton
from WootCharacter import WootCharacter
from WootGCN import WootSpatialGCN

import CSVHelper

import trimesh

class Runner:
    def __init__(self, conf, device = 'cuda'):
        self.conf = conf
        self.device = device

        #################################################################################################
        
        self.dof_dir = self.conf['dataset.skeleton_angles']
        self.dof_angle_normalized_dir = self.conf['dataset.skeleton_angles_rotation_normalized']

        #################################################################################################

        self.charactor = None
        # for embedded deoformations
        self.spatial_gcn = None
        # for per-vertex offset
        self.delta_gcn = None
        
        self.initialize_charactor()
        
        self.spatial_gcn.eval()
        self.delta_gcn.eval()

        #################################################################################################
      
        self.dof_arr = None
        self.dof_angle_normalized_arr = None     
        self.num_dof = -1
        self.tot_frame_num = -1

        self.load_dof()  
        
        #################################################################################################
        
        self.is_load_delta_checkpoints = self.conf.get_bool('train.load_delta_checkpoints', default=False)
        self.delta_checkpoint_dir = self.conf['train.delta_checkpoint_dir']
        
        if self.is_load_delta_checkpoints:
            self.load_delta_init_checkpoint()
            
        os.makedirs('mesh_output', exist_ok=True)
        
        return 

    def load_dof(self):
        print("+++++ Loading All Sorts of dofs")

        # here there is hack just for early end of the csv loader
        self.dof_arr = CSVHelper.load_csv_sequence_2D(
            self.dof_dir, type='float', skipRows=1, skipColumns=1
        )
        
        self.num_dof = self.dof_arr.shape[-1]
        self.tot_frame_num = self.dof_arr.shape[0]

        self.dof_angle_normalized_arr = (CSVHelper.load_csv_compact_4D(
            self.dof_angle_normalized_dir, 3, self.num_dof, 1, 1, 1, 'float'
        )).reshape((-1, 3, self.num_dof))
        
        print(
            ' dof shape: ', self.dof_arr.shape, '\n',
            'dof rotation normalized shape: ',self.dof_angle_normalized_arr.shape
        ) 
        print("+++++ Finished Loading All Sorts of dofs")
        return

    def load_delta_init_checkpoint(self):
        print('+++++ init with delta checkpoint', self.delta_checkpoint_dir)
        
        if os.path.isfile(self.delta_checkpoint_dir):
            cur_state_dict = torch.load(self.delta_checkpoint_dir, map_location=self.device)      
            if (self.spatial_gcn is not None) and ('spatial_gcn' in cur_state_dict.keys()):
                print('+++++ loading checkpoints spatial_gcn')
                self.spatial_gcn.load_state_dict(cur_state_dict['spatial_gcn'])     
            
            if (self.delta_gcn is not None) and ('delta_gcn' in cur_state_dict.keys()):
                print('+++++ loading checkpoints delta_gcn')
                self.delta_gcn.load_state_dict(cur_state_dict['delta_gcn'])             
        else:
            print(self.delta_checkpoint_dir, 'check point not found')
        
        return     

    def initialize_charactor(self):
        print('+++++ start initializing the character ')
        
        self.charactor = WootCharacter(
            **self.conf['character'],
            device=self.device
        )

        self.spatial_gcn = WootSpatialGCN(
            **self.conf['spatial_gcn'],
            obj_reader=self.charactor.graph_obj_reader, device=self.device
        )
        self.spatial_gcn = self.spatial_gcn.to(self.device)

        self.delta_gcn = WootSpatialGCN(
            **self.conf['delta_gcn'],
            obj_reader=self.charactor.obj_reader, device=self.device
        )
        self.delta_gcn = self.delta_gcn.to(self.device)
                
        print('+++++ end initializing the character ')
        return
    
    def gen_mesh_frames(self, frame_id):
        
        print('+++ generating: ', frame_id)

        history_frame_id = np.array([frame_id])
                            
        anglesNormalized0 = torch.FloatTensor(self.dof_angle_normalized_arr[history_frame_id][:,0,:]).to(self.device)
        anglesNormalized1 = torch.FloatTensor(self.dof_angle_normalized_arr[history_frame_id][:,1,:]).to(self.device)
        anglesNormalized2 = torch.FloatTensor(self.dof_angle_normalized_arr[history_frame_id][:,2,:]).to(self.device)
        
        concated_angles_normalized = torch.cat(
            [anglesNormalized0, anglesNormalized1, anglesNormalized2],  dim = 0
        )
        
        pose_only_template, picked_r, picked_t = self.charactor.compute_posed_template_embedded_graph(
            dof = concated_angles_normalized
        )
        
        v0 = pose_only_template[0:1, ...]
        v1 = pose_only_template[1:2, ...]
        v2 = pose_only_template[2:3, ...]

        inputTemporalPoseDeltaNet = torch.concat(
            [v0, v1, v2], dim = -1
        )

        r0 = picked_r[0:1, :, :]
        r1 = picked_r[1:2, :, :]
        r2 = picked_r[2:3, :, :]

        t0 = picked_t[0:1, :, :]
        t1 = picked_t[1:2 :, :]
        t2 = picked_t[2:3, :, :]

        inputTemporalPoseEGNet = torch.concat([
            t0 / 1000., r0, 
            t1 / 1000., r1, 
            t2 / 1000., r2
        ], dim = 2)

        #################################################################################################
        # for embedded deformation params
        eg_node_RT = self.spatial_gcn(inputTemporalPoseEGNet)
        
        delta_T = eg_node_RT[:, :, :3] * 1000.
        delta_R = eg_node_RT[:, :, 3:6]

        # for the per-vertex deformation
        per_vertex_deformation = self.delta_gcn(inputTemporalPoseDeltaNet)
        per_vertex_deformation = per_vertex_deformation * 1000.0    
  
        #################################################################################################
  
        dofs = self.dof_arr[history_frame_id,...]
        dofs = torch.FloatTensor(dofs.copy()).to(self.device)          

        ret_posed_template, ret_posed_eg, ret_posed_delta, org_template, eg_canoical, delta_canoical, joint_global_tranform = self.charactor.forward(
            dof = dofs, delta_R = delta_R, delta_T = delta_T, per_vertex_T = per_vertex_deformation
        )
        
        ret_verts = ret_posed_template[0].detach().cpu().numpy()

        #################################################################################################  
        
        trimesh.Trimesh(
            vertices = ret_verts,
            faces = self.charactor.temp_faces.cpu().numpy(),
            process= False
        ).export(os.path.join('mesh_output', str(frame_id) + '.ply'))
        
        #################################################################################################

        return ret_verts
    

if __name__ == '__main__':

    # Set up command line argument parser
    parser = ArgumentParser(description="Nothing To Say")
    parser.add_argument('--conf', type=str, default='./configs/test_config.conf')
    
    args = parser.parse_args(sys.argv[1:])

    f = open(args.conf)
    conf_text = f.read()
    f.close()
    preload_conf = ConfigFactory.parse_string(conf_text)

    runner = Runner(
        preload_conf
    )
    
    runner.gen_mesh_frames(600)
    
