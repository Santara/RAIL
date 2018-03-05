## To be run from inside the RAIL repository

#
# Initialisation

mkdir trajectories/training/Walker trajectories/evaluation/Walker
mkdir training_logs/Walker

#
# Training, following the GAIL and RAIL papers

python scripts/vis_mj.py expert_policies/modern/walker_eb5b2e_1.h5/snapshots/iter0000480 --count 25 --out trajectories/training/Walker/walker_expert_25.h5
python scripts/imitate_mj.py --mode ga --data trajectories/training/walker_expert_25.h5 --limit_trajs 25 --data_subsamp_freq 20 --env_name Walker-v1 --max_iter 501 --log training_logs/Walker/ga_501-iter_walker-GAIL.h5 
python scripts/imitate_mj.py --mode ga --data trajectories/training/walker_expert_25.h5 --limit_trajs 25 --data_subsamp_freq 20 --env_name Walker-v1 --max_iter 501 --useCVaR --CVaR_Lambda_not_trainable --CVaR_Lambda_val_if_not_trainable 0.25 --log training_logs/Walker/ga_501-iter_walker-RAIL.h5 

#
# Generating 50 trajectories each for evaluation

python scripts/vis_mj.py expert_policies/modern/walker_eb5b2e_1.h5/snapshots/iter0000480 --count 50 --out trajectories/evaluation/Walker/walker_expert_50.h5 	

python scripts/vis_mj.py training_logs/Walker/ga_501-iter_walker-GAIL_policy.h5/snapshots/iter0000500 --count 50 --out trajectories/evaluation/Walker/walker_GAIL_50.h5

python scripts/vis_mj.py training_logs/Walker/ga_501-iter_walker-RAIL_policy.h5/snapshots/iter0000500 --count 50 --out trajectories/evaluation/Walker/walker_RAIL_50.h5

#
# Plotting the overlayed training curves

python scripts/evaluate.py --trajectories_GAIL trajectories/evaluation/Walker/walker_GAIL_50.h5 --trajectories_RAIL trajectories/evaluation/Walker/walker_RAIL_50.h5 --env_name Walker --trajectories_expert trajectories/evaluation/Walker/walker_expert_50.h5