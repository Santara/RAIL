## To be run from inside the RAIL repository

#
# Initialisation

mkdir trajectories/training/Humanoid trajectories/evaluation/Humanoid
mkdir training_logs/Humanoid

#
# Training, following the GAIL and RAIL papers

python scripts/vis_mj.py expert_policies/modern/log_humanoid_1.h5/snapshots/iter0001940 --count 240 --out trajectories/training/Humanoid/humanoid_expert_240.h5
python scripts/imitate_mj.py --mode ga --data trajectories/training/humanoid_expert_240.h5 --limit_trajs 240 --data_subsamp_freq 20 --env_name Humanoid-v1 --max_iter 1501 --log training_logs/Humanoid/ga_1501-iter_humanoid-GAIL.h5 
python scripts/imitate_mj.py --mode ga --data trajectories/training/humanoid_expert_240.h5 --limit_trajs 240 --data_subsamp_freq 20 --env_name Humanoid-v1 --max_iter 1501 --useCVaR --CVaR_Lambda_not_trainable --CVaR_Lambda_val_if_not_trainable 0.75 --log training_logs/Humanoid/ga_1501-iter_humanoid-RAIL.h5 

#
# Generating 50 trajectories each for evaluation

python scripts/vis_mj.py expert_policies/modern/log_humanoid_1.h5/snapshots/iter0001940 --count 50 --out trajectories/evaluation/Humanoid/humanoid_expert_50.h5 	

python scripts/vis_mj.py training_logs/Humanoid/ga_1501-iter_humanoid-GAIL_policy.h5/snapshots/iter0001500 --count 50 --out trajectories/evaluation/Humanoid/humanoid_GAIL_50.h5

python scripts/vis_mj.py training_logs/Humanoid/ga_1501-iter_humanoid-RAIL_policy.h5/snapshots/iter0001500 --count 50 --out trajectories/evaluation/Humanoid/humanoid_RAIL_50.h5

#
# Plotting the overlayed training curves

python scripts/evaluate.py --trajectories_GAIL trajectories/evaluation/Humanoid/humanoid_GAIL_50.h5 --trajectories_RAIL trajectories/evaluation/Humanoid/humanoid_RAIL_50.h5 --env_name Humanoid --trajectories_expert trajectories/evaluation/Humanoid/humanoid_expert_50.h5