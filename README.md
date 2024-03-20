# Bayesian Sindy Autoencoders

Code for the paper **Bayesian autoencoders for data-driven discovery of coordinates, governing equations and fundamental constants** by Liyao Mars Gao and J. Nathan Kutz in Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences. This work is based on the prior work 'Data-driven discovery of coordinates and governing equations' PNAS, and the implementation is built upon Kathleen Champion's open-source software https://github.com/kpchamp/SindyAutoencoders. 

## Installation of related dependencies

Please refer to environments.yml for setup. For conda users, 

```bash
conda env create -f environment.yml -n bae
source activate bae
```

## How to run the code

We simplify the running into a simple script in each folder. For example, to run the training code for pendulum (real video), please go to folder examples/pendulum_real_video, and run the following line. 

```bash
sh sampling.sh
```

## Reaction-diffusion data

The Reaction-diffusion data we use is generated by MATLAB, and is too large to upload to Github. Please use the following link to download, and place it under examples/rd, or other places with route specified in example_reactiondiffusion.py file. To access the data, please use this link: https://drive.google.com/drive/folders/18DLuAp-nj5gI2U0-BLmTQcdaeveCdXHe?usp=sharing. 

## Raw pendulum video data

We open source the pendulum video data in this link: https://drive.google.com/file/d/1TilvyZg6VNNZ3CynO07BvBvsUXLRSaXh/view?usp=sharing. This data is collected from our lab with a GoPro camera. We encourage proper citation to this video and dataset for future usage. 

## Future works

We have a stronger version for GoPro physics ready for future works so please stay tuned. If you're interested in citing our work, please use the following for proper citation. 

```
@article{gao2022bayesian,
  title={Bayesian autoencoders for data-driven discovery of coordinates, governing equations and fundamental constants},
  author={Gao, L and Kutz, J Nathan},
  journal={arXiv preprint arXiv:2211.10575},
  year={2022}
}
```
