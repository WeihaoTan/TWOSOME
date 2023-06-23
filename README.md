# TWOSOME
Implementation of TWOSOME

# Installation
```
1. Create a conda environment
conda create -n twosome python=3.9
conda activate twosome

2. Install pytorch
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

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

# Run the code
```

# 1. For Tomato Salad environment
sh scripts/tomato_salad_ppo_llm.sh # train TWOSOME in Tomato Salad environment
sh scripts/tomato_salad_ppo.sh # train PPO in Tomato Salad environment

# 2. For Tomato Lettuce Salad environment
sh scripts/tomato_salad_lettuce_ppo_llm.sh # train TWOSOME in Tomato Lettuce Salad environment
sh scripts/tomato_salad_lettuce_ppo.sh # train PPO in Tomato Lettuce Salad environment

# 3. For Heat Pancake environment
sh scripts/heat_pancake_ppo_llm.sh # train TWOSOME in Heat Pancake environment
sh scripts/heat_pancake_ppo.sh # train PPO in Heat Pancake environment

# 4. For Watch TV environment
sh scripts/watch_tv_ppo_llm.sh # train TWOSOME in Watch TV environment
sh scripts/watch_tv_ppo.sh # train PPO in Watch TV environment

You can change the attribute, 'normalization-mode' in [sum, toekn, word], corresponding to TWOSOME without normalization, TWOSOME with token normalization and TWOSOME with word normalization

```

# ENV
Overcooked env is adapted from <https://github.com/WeihaoTan/gym-macro-overcooked>.
VirtualHome is adapted from <https://github.com/xavierpuigf/virtualhome>.
