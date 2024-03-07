# TWOSOME
Implementation of TWOSOME (True Knowledge Comes from Practice: Aligning LLMs with Embodied Environments via Reinforcement Learning). 

Arxiv link: https://arxiv.org/abs/2401.14151

<p align="center">
    <img src="GIFs/tomato_salad.gif" width=400></img>
    <img src="GIFs/tomato_lettuce_salad.gif" width=400></img>
</p>

<p align="center">
    <img src="GIFs/food_preparation.gif" width=400></img>
    <img src="GIFs/entertainment.gif" width=400></img>
</p>

# Installation
```
1. Create a conda environment
conda create -n twosome python=3.9
conda activate twosome

2. Install pytorch
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

3. Install requirements
pip install setuptools==65.5.0
pip install -r requirements.txt

4. Install overcooked environment
cd gym-macro-overcooked
pip install -e .
cd ..

5. Install virtual home environment
cd virtualhome
pip install -e .
```

# Train
```

# 1. For Tomato Salad environment
sh scripts/tomato_salad_ppo_llm.sh # train TWOSOME in Tomato Salad environment

# 2. For Tomato Lettuce Salad environment
sh scripts/tomato_salad_lettuce_ppo_llm.sh # train TWOSOME in Tomato Lettuce Salad environment

# 3. For Food Preparation environment
sh scripts/food_preparation_ppo_llm.sh # train TWOSOME in Food Preparation environment

# 4. For Entertainment environment
sh scripts/entertainment_ppo_llm.sh # train TWOSOME in Entertainment environment

You can change the attribute, 'normalization-mode' in [sum, toekn, word], corresponding to TWOSOME without normalization, TWOSOME with token normalization and TWOSOME with word normalization

```

# Inference
```
1. For food preparation environment
sh scripts/food_preparation_ppo_llm_inference.sh

2. For entertainment environment
sh scripts/entertainment_ppo_llm_inference.sh
```

# Environments
Overcooked env is adapted from <https://github.com/WeihaoTan/gym-macro-overcooked>.   
VirtualHome is adapted from <https://github.com/xavierpuigf/virtualhome>.

# Citation
If you find our work useful, please consider citing us!
```
@article{tan2024true,
  title={True Knowledge Comes from Practice: Aligning Large Language Models with Embodied Environments via Reinforcement Learning},
  author={Weihao Tan and Wentao Zhang and Shanqi Liu and Longtao Zheng and Xinrun Wang and Bo An},
  journal={arXiv preprint arXiv:2401.14151},
  year={2024}
}
```