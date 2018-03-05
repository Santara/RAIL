## To be run from inside the RAIL repository

#
# Initialisation

mkdir trajectories/training/Reacher trajectories/evaluation/Reacher
mkdir training_logs/Reacher

#
# Training, following the GAIL and RAIL papers

python scripts/vis_mj.py expert_policies/modern/log_Reacher-v1_4.h5/snapshots/iter0000500 --count 25 --out trajectories/training/Reacher/reacher_expert_18.h5
python scripts/imitate_mj.py --mode ga --data trajectories/training/Reacher/reacher_expert_18.h5 --limit_trajs 25 --data_subsamp_freq 20 --env_name Reacher-v1 --max_iter 201 --log training_logs/Reacher/ga_201-iter_reacher-GAIL.h5 
python scripts/imitate_mj.py --mode ga --data trajectories/training/Reacher/reacher_expert_18.h5 --limit_trajs 25 --data_subsamp_freq 20 --env_name Reacher-v1 --max_iter 201 --useCVaR --CVaR_Lambda_not_trainable --CVaR_Lambda_val_if_not_trainable 0.25 --log training_logs/Reacher/ga_201-iter_reacher-RAIL.h5 

#
# Generating 50 trajectories each for evaluation

python scripts/vis_mj.py expert_policies/modern/log_Reacher-v1_4.h5/snapshots/iter0000500 --count 50 --out trajectories/evaluation/Reacher/reacher_expert_50.h5 	

python scripts/vis_mj.py training_logs/Reacher/ga_201-iter_reacher-GAIL_policy.h5/snapshots/iter0000500 --count 50 --out trajectories/evaluation/Reacher/reacher_GAIL_50.h5

python scripts/vis_mj.py training_logs/Reacher/ga_201-iter_reacher-RAIL_policy.h5/snapshots/iter0000500 --count 50 --out trajectories/evaluation/Reacher/reacher_RAIL_50.h5

#
# Plotting the overlayed training curves

python scripts/evaluate.py --trajectories_GAIL trajectories/evaluation/Reacher/reacher_GAIL_50.h5 --trajectories_RAIL trajectories/evaluation/Reacher/reacher_RAIL_50.h5 --env_name Reacher --trajectories_expert trajectories/evaluation/Reacher/reacher_expert_50.h5