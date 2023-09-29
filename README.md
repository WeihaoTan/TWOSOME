# TWOSOME
Implementation of TWOSOME

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
sh scripts/tomato_salad_ppo.sh # train PPO in Tomato Salad environment

# 2. For Tomato Lettuce Salad environment
sh scripts/tomato_salad_lettuce_ppo_llm.sh # train TWOSOME in Tomato Lettuce Salad environment
sh scripts/tomato_salad_lettuce_ppo.sh # train PPO in Tomato Lettuce Salad environment

# 3. For Food Preparation environment
sh scripts/food_preparation_ppo_llm.sh # train TWOSOME in Food Preparation environment
sh scripts/food_preparation_ppo.sh # train PPO in Food Preparation environment

# 4. For Entertainment environment
sh scripts/entertainment_ppo_llm.sh # train TWOSOME in Entertainment environment
sh scripts/entertainment_ppo.sh # train PPO in Entertainment environment

You can change the attribute, 'normalization-mode' in [sum, toekn, word], corresponding to TWOSOME without normalization, TWOSOME with token normalization and TWOSOME with word normalization

```

# Inference
```
1. For food preparation environment
sh scripts/food_preparation_ppo_llm_inference.sh

2. For entertainment environment
sh scripts/entertainment_ppo_llm_inference.sh
```

# ENV
Overcooked env is adapted from <https://github.com/WeihaoTan/gym-macro-overcooked>.   
VirtualHome is adapted from <https://github.com/xavierpuigf/virtualhome>.
