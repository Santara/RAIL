python scripts/imitate_mj.py --mode ga --data trajectories/expert_trajectories-Humanoid --limit_trajs 250 --data_subsamp_freq 20 --env_name Humanoid-v1 --log training_logs/additiveStatePrior/Test/conditionalThrashing_b20_kt80_keepKicking.h5 --resume_training --appendFlag --checkpoint training_logs/additiveStatePrior/Test/conditionalThrashing_b20_kt80_keepKicking.h5/snapshots/iter0000260 --max_iter 1501  --use_additiveStatePrior --familiarity_beta 20 --familiarity_alpha 100000 --kickThreshold_percentile 80.0 --use_expert_traj_filtering --expert_traj_filt_percentile_threshold 10
