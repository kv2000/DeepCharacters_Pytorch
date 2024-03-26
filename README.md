# Deep Characters Pytorch 
Pytorch implementation of the components in <a href="https://people.mpi-inf.mpg.de/~mhaberma/projects/2021-ddc/"><strong>Realtime Deep Dynamic Characters</strong></a> (Habermann et. al., Siggraph 2021)

---

# News
**2024-3-29** The inital version **Deep Characters Pytorch** is available, including an minimal runnable version of the <strong><font color=red>skeleton</font></strong> and the <strong><font color=blue>embedded deformed clothed human character</font></strong> for the <a href="https://people.mpi-inf.mpg.de/~mhaberma/projects/2021-ddc/"><strong>DynaCap</strong></a>, <a href="https://people.mpi-inf.mpg.de/~mhaberma/projects/2021-ddc/"><strong>TriHuman</strong></a>, and <a href="https://people.mpi-inf.mpg.de/~mhaberma/projects/2021-ddc/"><strong>ASH</strong></a> dataset. Moreover, we provide <strong><font color=green>GCN-based deformation network</font></strong> mentioned in the paper.

---
# Installation

### Requirements

```bash
conda create -n mpiiddc python=3.9
conda activate mpiiddc

# tested to work on this combination
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install -c conda-forge cudatoolkit-dev=11.3.1

# download and install pytorch3d==0.7.1
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
git checkout 995b60e
python setup.py install

# install other dependcies
pip install -r requirements.txt
```

### Build The Customized Components
```bash
cd cuda_skeleton
TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6" python setup.py install
```

---

# Examples
### Testing the Skeleton
After initialization, the skeleton funcs takes a batch of **DOFs** and produces the joint transformations. The following script will dump joints as point clouds.

```bash
cd examples
python 0_test_skeleton.py
```
### Testing the Deformable Character
After initialization, the deformable character funcs takes a batch of **DOFs**, **Embedded Deformation Params**, and **Per-vertex Offset**,  and produces the posed/unposed, <strong><font color=red>template</font></strong>, <strong><font color=blue>embedded deformed template</font></strong>, <strong><font color=green>embedded deformed template with per-vertex offsets</font></strong>.

Here, the **Embedded Deformation Params**, and **Per-vertex Offset** is set to 0 by default.
Moreover, you can switch between **lbs/dqs**, and wheather activate **embedded deformation** in ``./examples/test_config.conf``. 

```bash
cd examples
python 1_test_character.py
```

### Testing the Character with learnable deformations
Also we can apply the learnable deformation learned with SpatialGNN. You may download the weights  <a href=""><strong>Here</strong></a>, extract and put the checkpoint under ```./examples/checkpoints/``` folder. 
Then we can generate the **posed/unposed character with learned defomration** through running the following blocks. 

```bash
cd examples
python 2_test_deformable_character.py
```

---

# Todo

- [x] Example for skeleton
- [x] Example for the deformable character
- [x] Example for the learnable embeeded deformable character
- [ ] More to come

---

# Citation

If you find our work useful for your research, please, please, and please, consider citing <a href="https://vcai.mpi-inf.mpg.de/projects/trihuman/"><strong>TriHuman</strong></a> (where I build this repo for :D), and also the original paper <a href="https://people.mpi-inf.mpg.de/~mhaberma/projects/2021-ddc/"><strong>Real-time Deep Dynamic Characters</strong></a>.

```
@article{habermann2021,
	author = {Marc Habermann and Lingjie Liu and Weipeng Xu and Michael Zollhoefer and Gerard Pons-Moll and Christian Theobalt},
	title = {Real-time Deep Dynamic Characters},
	journal = {ACM Transactions on Graphics}, 
	month = {aug},
	volume = {40},
	number = {4}, 
	articleno = {94},
	year = {2021}, 
	publisher = {ACM}
} 

```

```
@misc{zhu2023trihuman,
    title={TriHuman : A Real-time and Controllable Tri-plane Representation for Detailed Human Geometry and Appearance Synthesis}, 
    author={Heming Zhu and Fangneng Zhan and Christian Theobalt and Marc Habermann},
    year={2023},
    eprint={2312.05161},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

