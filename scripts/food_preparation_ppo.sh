CUDA_VISIBLE_DEVICES=0 \
python twosome/virtualhome/ppo_raw_env_v1.py \
--exp-name "heat_pancake_ppo" \
--learning-rate 1e-3 \
--num-envs 4 \
--num-steps 128 \
--total-timesteps 50000 \
--env-id "VirtualHome-v1"