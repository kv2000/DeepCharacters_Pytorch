"""
@File: character_utils.py
@Author: Heming Zhu
@Email: hezhu@mpi-inf.mpg.de
@Date: 2023-09-25
@Desc: The character utils. some translated/imspired from/by the original c++ code.
"""

import torch
import kornia
import math
import numpy as np

def dual_quad_to_trans_vec(_quat_0, _quat_e):

    t_x = 2.*(-_quat_e[...,0]* _quat_0[...,1] + _quat_e[...,1] * _quat_0[...,0] - _quat_e[...,2]*_quat_0[...,3] + _quat_e[...,3]*_quat_0[...,2])
    t_y = 2.*(-_quat_e[...,0]* _quat_0[...,2] + _quat_e[...,1] * _quat_0[...,3] + _quat_e[...,2]*_quat_0[...,0] - _quat_e[...,3]*_quat_0[...,1]) 
    t_z = 2.*(-_quat_e[...,0]* _quat_0[...,3] - _quat_e[...,1] * _quat_0[...,2] + _quat_e[...,2]*_quat_0[...,1] + _quat_e[...,3]*_quat_0[...,0]) 

    return torch.stack(
        [t_x, t_y, t_z], dim = -1
    )

def wrap_transformation_angle(org_tansformation_mat):
    print(org_tansformation_mat.shape)

    rotation_mat = org_tansformation_mat[:,:3,:3]
    translation_mat = org_tansformation_mat[:,:3,3:]
    
    rotation_vec = kornia.geometry.conversions.rotation_matrix_to_angle_axis(rotation_mat.contiguous())

    rotation_axis = torch.nn.functional.normalize(rotation_vec, dim = -1)

    rotation_angle = (
        rotation_vec / rotation_axis
    )[...,0]
    
    pi_tensor = torch.ones_like(rotation_angle, device=rotation_angle.device) * math.pi
    rotation_angle = ((rotation_angle + pi_tensor) % (2 * pi_tensor)) - pi_tensor
    rotation_angle[rotation_angle.eq(-pi_tensor)] = math.pi

    updated_rotation_vec = rotation_axis * rotation_angle[...,None]

    updated_rotation_mat = kornia.geometry.conversions.angle_axis_to_rotation_matrix(updated_rotation_vec)

    return kornia.geometry.conversions.Rt_to_matrix4x4(
        updated_rotation_mat, translation_mat
    )

def compute_geodesic_distance(st_vert, neighbor_idx, num_verts, distance_limit = 1145141919):
    
    q_q = [st_vert]
    
    ret_dist = [1145141919 for i in range(num_verts)]
    ret_dist[st_vert] = 0
    in_the_queue = [False for i in range(num_verts)]
    in_the_queue[st_vert] = True

    while(len(q_q) > 0):    
        cur_id = q_q[0]
        q_q.pop(0)
        
        for next_id in neighbor_idx[cur_id]:
            if ((ret_dist[cur_id] + 1) < ret_dist[next_id]):
                ret_dist[next_id] = ret_dist[cur_id] + 1
                if not in_the_queue[next_id]:
                    q_q.append(next_id)
                    in_the_queue[next_id] = True

        in_the_queue[cur_id] = False

    return ret_dist

def compute_trans_quad(q, t):
    # q -> wxyz order
    # q:  0 -> w, 1 - > i, 2 -> j, 3 - > k
    
    w = -0.5 * ( t[...,0] * q[...,1] + t[...,1] * q[...,2] + t[...,2] * q[...,3] )
    i = 0.5  * ( t[...,0] * q[...,0] + t[...,1] * q[...,3] - t[...,2] * q[...,2] )
    j = 0.5  * (-t[...,0] * q[...,3] + t[...,1] * q[...,0] + t[...,2] * q[...,1] )
    k = 0.5  * ( t[...,0] * q[...,2] - t[...,1] * q[...,1] + t[...,2] * q[...,0] )

    # there should be some normalizations
    return torch.stack([w, i, j, k], dim=-1)

def batch_angle_to_mat(angles):
    batch_size = angles.shape[0]
    
    cosAlpha	= torch.cos(angles[...,0])
    cosBeta		= torch.cos(angles[...,1])
    cosGamma	= torch.cos(angles[...,2])
    sinAlpha	= torch.sin(angles[...,0]) 
    sinBeta		= torch.sin(angles[...,1])
    sinGamma	= torch.sin(angles[...,2])
    
    r_00 = cosBeta * cosGamma
    r_01 = sinAlpha * sinBeta * cosGamma - cosAlpha * sinGamma
    r_02 = cosAlpha * sinBeta * cosGamma + sinAlpha * sinGamma
    
    r_10 = cosBeta * sinGamma
    r_11 = sinAlpha * sinBeta * sinGamma + cosAlpha * cosGamma
    r_12 = cosAlpha * sinBeta * sinGamma - sinAlpha * cosGamma

    r_20 = -sinBeta
    r_21 = sinAlpha * cosBeta
    r_22 = cosAlpha * cosBeta

    ret_mat = torch.stack(
        [r_00, r_01, r_02, r_10, r_11, r_12, r_20, r_21, r_22], dim = -1
    ).reshape([batch_size, -1, 3, 3])

    return ret_mat

def batch_mat_to_angle(r_mat):
    # compute singular 
    sy = torch.sqrt(
        r_mat[:,0,0] * r_mat[:,0,0] + r_mat[:,1,0] * r_mat[:,1,0]
    )
    
    non_singluar = (sy >= 1e-6)
    
    if torch.all(non_singluar):

        e_x_0 = torch.atan2(r_mat[:,2,1], r_mat[:,2,2])
        e_y_0 = torch.atan2(-r_mat[:,2,0], sy)
        e_z_0 = torch.atan2(r_mat[:,1,0], r_mat[:,0,0])

        ret_angle = torch.stack(
            [e_x_0, e_y_0, e_z_0], dim = -1
        )
    
    else:
        e_x_0 = torch.atan2(r_mat[:,2,1], r_mat[:,2,2])
        e_y_0 = torch.atan2(-r_mat[:,2,0], sy)
        e_z_0 = torch.atan2(r_mat[:,1,0], r_mat[:,0,0])
        
        ret_angle_0 = torch.stack(
            [e_x_0, e_y_0, e_z_0], dim = -1
        )
        
        e_x_1 = torch.atan2(-r_mat[:,1,2], r_mat[:,1,1])
        e_y_1 = torch.atan2(-r_mat[:,2,0], sy)
        e_z_1 = torch.zeros_like(e_y_1,device=e_y_1.device)

        ret_angle_1 = torch.stack(
            [e_x_1, e_y_1, e_z_1], dim = -1
        )

        is_singular = torch.logical_not(non_singluar).float()
        non_singluar = non_singluar.float()
        ret_angle = ret_angle_0 * non_singluar + ret_angle_1 * is_singular

    return ret_angle