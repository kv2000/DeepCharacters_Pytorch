"""
@File: test_skeleton.py
@Author: Heming Zhu
@Email: hezhu@mpi-inf.mpg.de
@Date: 2024-03-25
@Desc: Some test scripts for the character skeleton.
"""
import sys
sys.path.append("../")

import os
from pyhocon import ConfigFactory
from argparse import ArgumentParser, Namespace

import torch

from WootSkeleton import WootSkeleton
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
    temp_skeleton = WootSkeleton(
        skeleton_dir= preload_conf['character']['skeleton_dir'],
        device = 'cuda'
    )

    # load the dofs, remove the end_frame entry to read all
    dof_arr = CSVHelper.load_csv_sequence_2D(
        preload_conf['training']['skeleton_angles'], type='float', skipRows=1, skipColumns=1 #, end_frame=(1000)
    )
        
    #################################################################################################
    
    # b X (joint/dof num) X 4 X 4
    ret_joint_transformation, ret_local_joint_translation, ret_local_joint_transformation = temp_skeleton.forward(
        torch.FloatTensor(dof_arr[:1,:]).to('cuda')
    )
    
    ret_joint_postion = ret_joint_transformation[:,:,:3,3]
    
    """
    # dump the joint/dof positions,
    import trimesh
    trimesh.Trimesh(
        vertices=ret_joint_postion[0].detach().cpu().numpy()
    ).export('test.ply')
    """
    
