## To be run from inside the RAIL repository

#
# Initialisation

mkdir trajectories/training/HalfCheetah trajectories/evaluation/HalfCheetah
mkdir training_logs/HalfCheetah

#
# Training, following the GAIL and RAIL papers

# Make the dataset of expert demonstrations
python scripts/vis_mj.py expert_policies/modern/log_HalfCheetah-v1_2.h5/snapshots/iter0000500 --count 25 --out trajectories/training/HalfCheetah/halfcheetah_expert_25.h5
# Train GAIL agent
python scripts/imitate_mj.py --mode ga --data trajectories/training/halfcheetah_expert_25.h5 --limit_trajs 25 --data_subsamp_freq 20 --env_name HalfCheetah-v1 --max_iter 501 --log training_logs/HalfCheetah/ga_501-iter_halfcheetah-GAIL.h5 
# Train RAIL agent
python scripts/imitate_mj.py --mode ga --data trajectories/training/halfcheetah_expert_25.h5 --limit_trajs 25 --data_subsamp_freq 20 --env_name HalfCheetah-v1 --max_iter 501 --useCVaR --CVaR_Lambda_not_trainable --CVaR_Lambda_val_if_not_trainable 0.5 --log training_logs/HalfCheetah/ga_501-iter_halfcheetah-RAIL.h5 

#
# Generating 50 trajectories each for evaluation

python scripts/vis_mj.py expert_policies/modern/log_HalfCheetah-v1_2.h5/snapshots/iter0000500 --count 50 --out trajectories/evaluation/HalfCheetah/halfcheetah_expert_50.h5 	

python scripts/vis_mj.py training_logs/HalfCheetah/ga_501-iter_halfcheetah-GAIL_policy.h5/snapshots/iter0000500 --count 50 --out trajectories/evaluation/HalfCheetah/halfcheetah_GAIL_50.h5

python scripts/vis_mj.py training_logs/HalfCheetah/ga_501-iter_halfcheetah-RAIL_policy.h5/snapshots/iter0000500 --count 50 --out trajectories/evaluation/HalfCheetah/halfcheetah_RAIL_50.h5

#
# Plotting the overlayed training curves

python scripts/evaluate.py --trajectories_GAIL trajectories/HalfCheetah/halfcheetah_GAIL_50.h5 --trajectories_RAIL trajectories/HalfCheetah/halfcheetah_RAIL_50.h5 --env_name HalfCheetah --trajectories_expert trajectories/HalfCheetah/halfcheetah_expert_250.h5