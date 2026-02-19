#!/bin/bash
#BSUB -q hpc
#BSUB -J "dual_nyc_man_south_paper[1-2]"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 72:00
#BSUB -o logs/dual_nyc_man_south_paper_mode%I_%J.out
#BSUB -e logs/dual_nyc_man_south_paper_mode%I_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

# Subtract 1 from job index to get mode 0-2
MODE=$((LSB_JOBINDEX - 1))

python main_a2c_multi_agent.py  --reward_scalar 2000 --critic_warmup_episodes 50 --mode $MODE --city "nyc_man_south" --q_lr 0.0004 --p_lr 0.0002 --actor_clip 1000 --critic_clip 1000 --max_episodes 150000 --od_price_observe --od_price_actions --checkpoint_path dual_nyc_man_south_paper_mode${MODE}