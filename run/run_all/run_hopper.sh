## To be run from inside the RAIL repository

#
# Initialisation

mkdir trajectories/training/Hopper trajectories/evaluation/Hopper 
mkdir training_logs/Hopper

#
# Training, following the GAIL and RAIL papers

# Make the dataset of expert demonstrations
python scripts/vis_mj.py expert_policies/modern/log_Hopper-v1_4.h5/snapshots/iter0000500 --count 25 --out trajectories/training/Hopper/hopper_expert_25.h5
# Train GAIL agent
python scripts/imitate_mj.py --mode ga --data trajectories/training/Hopper/hopper_expert_25.h5 --limit_trajs 25 --data_subsamp_freq 20 --env_name Hopper-v1 --max_iter 501 --log training_logs/Hopper/ga_501-iter_hopper-GAIL.h5
# Train RAIL agent
python scripts/imitate_mj.py --mode ga --data trajectories/training/Hopper/hopper_expert_25.h5 --limit_trajs 25 --data_subsamp_freq 20 --env_name Hopper-v1 --max_iter 501 --useCVaR --CVaR_Lambda_not_trainable --CVaR_Lambda_val_if_not_trainable 0.5 --log training_logs/Hopper/ga_501-iter_hopper-RAIL.h5

#
# Generating 50 trajectories each for evaluation

python scripts/vis_mj.py expert_policies/modern/log_Hopper-v1_4.h5/snapshots/iter0000500 --count 50 --out trajectories/evaluation/Hopper/hopper_expert_50.h5 	

python scripts/vis_mj.py training_logs/Hopper/ga_501-iter_hopper-GAIL_policy.h5/snapshots/iter0000500 --count 50 --out trajectories/evaluation/Hopper/hopper_GAIL_50.h5

python scripts/vis_mj.py training_logs/Hopper/ga_501-iter_hopper-RAIL_policy.h5/snapshots/iter0000500 --count 50 --out trajectories/evaluation/Hopper/hopper_RAIL_50.h5

#
# Plotting the overlayed training curves

python scripts/evaluate.py --trajectories_GAIL trajectories/evaluation/Hopper/hopper_GAIL_50.h5 --trajectories_RAIL trajectories/evaluation/Hopper/hopper_RAIL_50.h5 --env_name Hopper --trajectories_expert trajectories/evaluation/Hopper/hopper_expert_50.h5