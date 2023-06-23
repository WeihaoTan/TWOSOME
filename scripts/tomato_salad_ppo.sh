CUDA_VISIBLE_DEVICES=0 \
python twosome/overcooked/ppo_raw_env.py \
--exp-name "tomato_salad_ppo" \
--learning-rate 1e-4 \
--num-envs 4 \
--num-steps 128 \
--total-timesteps 500000 \
--task 0 \
--env-id "Overcooked-LLMA-v4"