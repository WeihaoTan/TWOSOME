CUDA_VISIBLE_DEVICES=0 \
python twosome/virtualhome/ppo_raw_env_v2.py \
--exp-name "watch_tv_ppo" \
--learning-rate 1e-3 \
--num-envs 4 \
--num-steps 128 \
--total-timesteps 200000 \
--env-id "VirtualHome-v2"