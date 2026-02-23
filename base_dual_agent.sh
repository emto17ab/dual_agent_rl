#!/bin/bash
#BSUB -q hpc
#BSUB -J "dual_washington_dc_paper_v11[3]"
#BSUB -n 3
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 72:00
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -o logs/dual_washington_dc_paper_v11_mode%I_%J.out
#BSUB -e logs/dual_washington_dc_paper_v11_mode%I_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

# Subtract 1 from job index to get mode 0-2
MODE=$((LSB_JOBINDEX - 1))

python main_a2c_multi_agent.py --reward_scalar 4000 --critic_warmup_episodes 50 --mode $MODE --city "washington_dc" --q_lr 0.0004 --p_lr 0.0002 --actor_clip 1000 --critic_clip 1000 --max_episodes 150000 --od_price_observe --checkpoint_path dual_washington_dc_paper_v11_mode${MODE}