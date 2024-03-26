"""
@File: 1_test_character.py
@Author: Heming Zhu
@Email: hezhu@mpi-inf.mpg.de
@Date: 2024-03-25
@Desc: Some test scripts for the character.
"""
import sys
sys.path.append("../")
sys.path.append("../../")

import os
from pyhocon import ConfigFactory
from argparse import ArgumentParser, Namespace

import torch

from WootSkeleton import WootSkeleton
from WootCharacter import WootCharacter

import CSVHelper

if __name__ == '__main__':
        
    # Set up command line argument parser
    parser = ArgumentParser(description="Nothing To Say")
    parser.add_argument('--conf', type=str, default='./configs/test_config.conf')
    
    args = parser.parse_args(sys.argv[1:])

    f = open(args.conf)
    conf_text = f.read()
    f.close()
    preload_conf = ConfigFactory.parse_string(conf_text)
    
    #################################################################################################
    
    # create the skeleton 
    test_charactor = WootCharacter(
        **preload_conf['character'], 
        device='cuda'
    )

    # load the dofs, remove the end_frame entry to read all
    dof_arr = CSVHelper.load_csv_sequence_2D(
        preload_conf['dataset']['skeleton_angles'], type='float', skipRows=1, skipColumns=1 #, end_frame=(1000)
    )
        
    #################################################################################################
    # if delta_R, delta_T is set None, then no non-rigid deformation
    # if per_vertex_T is set None, then no non-rigid deformation
    posed_org, posed_eg, posed_delta, org_canonical, eg_canoical, delta_canoical, ret_global_tranform = test_charactor.forward(
        dof = torch.FloatTensor(dof_arr[500:600]).to('cuda'),
        delta_R=None, delta_T=None, per_vertex_T=None
    )
    
    # duuump 
    import trimesh
    
    trimesh.Trimesh(
        vertices = posed_delta[0].detach().cpu().numpy(), 
        faces = test_charactor.temp_faces.cpu().numpy(), 
        process= False
    ).export('posed_delta.ply')

    trimesh.Trimesh(
        vertices = delta_canoical[0].detach().cpu().numpy(), 
        faces = test_charactor.temp_faces.cpu().numpy(), 
        process= False
    ).export('delta_canoical.ply')
    
    trimesh.Trimesh(
        vertices=ret_global_tranform[0,:,:3,3].detach().cpu().numpy()
    ).export('joint_position.ply')
    
    
