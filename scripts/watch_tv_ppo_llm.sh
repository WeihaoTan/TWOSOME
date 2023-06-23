CUDA_VISIBLE_DEVICES=0 \
python twosome/virtualhome/ppo_llm_v2.py \
  --exp-name "watch_tv_ppo_llm"\
  --policy-learning-rate 1e-6 \
  --value-learning-rate 5e-5 \
  --num-envs 4 \
  --num-steps 32 \
  --policy-num-minibatches 32 \
  --value-num-minibatches 4 \
  --update-epochs 1 \
  --total-timesteps 200000 \
  --critic-warm-up-steps 0 \
  --target-kl 0.02 \
  --gradient-checkpointing-steps 8 \
  --env-id "VirtualHome-v2" \
  --record-path "workdir" \
  --normalization-mode "word" \
  --gamma 0.95