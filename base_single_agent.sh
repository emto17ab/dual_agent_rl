#!/bin/bash
#BSUB -q hpc
#BSUB -J "single_agent_nyc_man_south_paper_v2[1-3]"
#BSUB -n 3
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 72:00
#BSUB -o logs/single_agent_nyc_man_south_paper_v2_mode%I_%J.out
#BSUB -e logs/single_agent_nyc_man_south_paper_v2_mode%I_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

# Subtract 1 from job index to get mode 0-2
MODE=$((LSB_JOBINDEX - 1))

python main_a2c.py --reward_scalar 4000 --critic_warmup_episodes 50 --mode ${MODE} --city "nyc_man_south" --q_lr 0.0004 --p_lr 0.0002 --actor_clip 1000 --critic_clip 1000 --max_episodes 150000 --od_price_observe --checkpoint_path single_agent_nyc_man_south_paper_v2_mode${MODE}