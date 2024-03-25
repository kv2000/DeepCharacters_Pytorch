/*
@File: cuda_skeleton.cpp
@Author: Heming Zhu
@Email: hezhu@mpi-inf.mpg.de
@Date: 2023-09-25
@Desc: The kernel for building the pytorch cuda skeleton.
*/

#include <torch/extension.h>
#include <iostream>
#include <math.h> 
// cuda render utils

using namespace std;
using namespace torch::indexing;

torch::Tensor skeleton_fw(
    int number_of_joints,
    std::vector<torch::Tensor> current_id_arr,
    std::vector<torch::Tensor> parent_id_arr,
    torch::Tensor ret_local_joint_transformation
){
    int number_of_batches = ret_local_joint_transformation.size(0);
    int number_of_layers = current_id_arr.size();
    auto options = ret_local_joint_transformation.options().dtype(torch::kFloat32).device(ret_local_joint_transformation.device()).layout(torch::kStrided).requires_grad(true);
    auto ret_joint_transformation = torch::eye(4, options).expand({number_of_batches, number_of_joints, 4, 4}).contiguous();
    
    for (int i = 0; i < number_of_layers; i++) {
        
        torch::Tensor cur_parent = parent_id_arr[i];
        torch::Tensor cur_idx = current_id_arr[i];
        
        auto cur_transformation = ret_joint_transformation.index({Slice(), cur_parent, "..."}).matmul(
            ret_local_joint_transformation.index({Slice(), cur_idx, "..."})
        );

        ret_joint_transformation.index_put_({Slice(), cur_idx, "..."}, cur_transformation);
    }

    return ret_joint_transformation;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("skeleton_fw", &skeleton_fw);
}