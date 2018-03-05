import argparse
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.manifold import TSNE 
from sklearn.mixture import GaussianMixture as GMM
from environments import rlgymenv
import gym
from time import time 
import policyopt
from policyopt import SimConfig, rl, util, nn, tqdm
from scripts import imitate_mj

import os.path
import pdb

TINY_ARCHITECTURE = '[{"type": "fc", "n": 64}, {"type": "nonlin", "func": "tanh"}, {"type": "fc", "n": 64}, {"type": "nonlin", "func": "tanh"}]'
SIMPLE_ARCHITECTURE = '[{"type": "fc", "n": 100}, {"type": "nonlin", "func": "tanh"}, {"type": "fc", "n": 100}, {"type": "nonlin", "func": "tanh"}]'


def load_dataset(filename
	# , limit_trajs, data_subsamp_freq
	, len_filtering=False, len_filter_threshold=10
	):
    # Load expert data
    with h5py.File(filename, 'r') as f:
        # Read data as written by vis_mj.py
        dset_size = full_dset_size = f['obs_B_T_Do'].shape[0] # full dataset size
        # dset_size = min(full_dset_size, limit_trajs) if limit_trajs is not None else full_dset_size

        exobs_B_T_Do = f['obs_B_T_Do'][:dset_size,...][...]
        exa_B_T_Da = f['a_B_T_Da'][:dset_size,...][...]
        exr_B_T = f['r_B_T'][:dset_size,...][...] #Total reward obtained from each trajectory
        exlen_B = f['len_B'][:dset_size,...][...] #Length of trajectories

    if len_filtering:
        len_thresh = np.percentile(exlen_B,len_filter_threshold)
        filtered_indices = exlen_B>=len_thresh

        exobs_B_T_Do = exobs_B_T_Do[filtered_indices]
        exa_B_T_Da = exa_B_T_Da[filtered_indices]
        exr_B_T = exr_B_T[filtered_indices]
        exlen_B = exlen_B[filtered_indices]


    print 'Dataset size: {} transitions ({} trajectories)'.format(exlen_B.sum(), len(exlen_B))
    print 'Average return:', exr_B_T.sum(axis=1).mean()

    # # Stack everything together
    # start_times_B = np.random.RandomState(0).randint(0, data_subsamp_freq, size=exlen_B.shape[0]) 
    # print 'start times'
    # print start_times_B
    # exobs_Bstacked_Do = np.concatenate(
    #     [exobs_B_T_Do[i,  start_times_B[i] : l : data_subsamp_freq,  :] for i, l in enumerate(exlen_B)],
    #     axis=0)
    # exa_Bstacked_Da = np.concatenate(
    #     [exa_B_T_Da[i,start_times_B[i]:l:data_subsamp_freq,:] for i, l in enumerate(exlen_B)],
    #     axis=0)
    # ext_Bstacked = np.concatenate(
    #     [np.arange(start_times_B[i], l, step=data_subsamp_freq) for i, l in enumerate(exlen_B)]).astype(float)

    # assert exobs_Bstacked_Do.shape[0] == exa_Bstacked_Da.shape[0] == ext_Bstacked.shape[0]# == np.ceil(exlen_B.astype(float)/data_subsamp_freq).astype(int).sum() > 0

    # print 'Subsampled data every {} timestep(s)'.format(data_subsamp_freq)
    # print 'Final dataset size: {} transitions (average {} per traj)'.format(exobs_Bstacked_Do.shape[0], float(exobs_Bstacked_Do.shape[0])/dset_size)

    # return exobs_Bstacked_Do, exa_Bstacked_Da, ext_Bstacked
    return exobs_B_T_Do, exa_B_T_Da, exr_B_T, exlen_B





def find_deviation_of_agent_actions_from_expert_actions_for_observations_from_expert_trajectories(expert_trajectories, learner_policy, limit_trajs, data_subsamp_freq, ipython_after_eval):
	# Load the learner's policy
	policy_file, policy_key = util.split_h5_name(learner_policy)
	print 'Loading policy parameters from %s in %s' % (policy_key, policy_file)
	with h5py.File(policy_file, 'r') as f:
	    train_args = json.loads(f.attrs['args'])
	    dset = f[policy_key]
	    import pprint
	    pprint.pprint(dict(dset.attrs))

	# Initialize the MDP
	env_name = train_args['env_name']
	print 'Loading environment', env_name
	mdp = rlgymenv.RLGymMDP(env_name)
	util.header('MDP observation space, action space sizes: %d, %d\n' % (mdp.obs_space.dim, mdp.action_space.storage_size))

	# Initialize the policy and load its parameters
	enable_obsnorm = bool(train_args['enable_obsnorm']) if 'enable_obsnorm' in train_args else train_args['obsnorm_mode'] != 'none'
	if isinstance(mdp.action_space, policyopt.ContinuousSpace):
	    policy_cfg = rl.GaussianPolicyConfig(
	        hidden_spec=train_args['policy_hidden_spec'],
	        min_stdev=0.,
	        init_logstdev=0.,
	        enable_obsnorm=enable_obsnorm)
	    policy = rl.GaussianPolicy(policy_cfg, mdp.obs_space, mdp.action_space, 'GaussianPolicy')
	else:
	    policy_cfg = rl.GibbsPolicyConfig(
	        hidden_spec=train_args['policy_hidden_spec'],
	        enable_obsnorm=enable_obsnorm)
	    policy = rl.GibbsPolicy(policy_cfg, mdp.obs_space, mdp.action_space, 'GibbsPolicy')

	policy.load_h5(policy_file, policy_key)

	# Load the expert trajectories
	exobs_Bstacked_Do, exa_Bstacked_Da, ext_Bstacked = imitate_mj.load_dataset(
	    expert_trajectories, limit_trajs, data_subsamp_freq)
	assert exobs_Bstacked_Do.shape[1] == mdp.obs_space.storage_size
	assert exa_Bstacked_Da.shape[1] == mdp.action_space.storage_size
	assert ext_Bstacked.ndim == 1



	# Generate the actions according to the learner's policy for the expert's observations
	learner_actions_Bstacked_Da = policy.sample_actions(exobs_Bstacked_Do)[0]

	# Calcualating the deviation histogram:
	action_deviations = np.linalg.norm(exa_Bstacked_Da - learner_actions_Bstacked_Da, axis=1)

	# Plot the histogram
	# sns.kdeplot(action_deviations,shade=True)

	# FIXME: Uncomment the following
	plt.figure()
	plt.hist(action_deviations, bins=100)
	plt.savefig('deviation_of_agent_actions_from_expert_actions_for_observations_from_expert_trajectories.png')
	plt.show()	

	if ipython_after_eval:
		import IPython; IPython.embed()    

def find_deviation_of_agent_actions_from_expert_actions_for_underperforming_trajectories(learner_trajectories, expert_policy, lower_bound_reward, ipython_after_eval, generate_plot):
	obs,a,r,l = find_underperforming_trajectories(learner_trajectories, lower_bound_reward)
	print(type(obs))
	# Load the expert's policy
	policy_file, policy_key = util.split_h5_name(expert_policy)
	print 'Loading policy parameters from %s in %s' % (policy_key, policy_file)
	with h5py.File(policy_file, 'r') as f:
	    train_args = json.loads(f.attrs['args'])
	    dset = f[policy_key]
	    import pprint
	    pprint.pprint(dict(dset.attrs))

	# Initialize the MDP
	env_name = train_args['env_name']
	print 'Loading environment', env_name
	mdp = rlgymenv.RLGymMDP(env_name)
	util.header('MDP observation space, action space sizes: %d, %d\n' % (mdp.obs_space.dim, mdp.action_space.storage_size))

	# Initialize the policy and load its parameters
	enable_obsnorm = bool(train_args['enable_obsnorm']) if 'enable_obsnorm' in train_args else train_args['obsnorm_mode'] != 'none'
	if isinstance(mdp.action_space, policyopt.ContinuousSpace):
	    policy_cfg = rl.GaussianPolicyConfig(
	        hidden_spec=train_args['policy_hidden_spec'],
	        min_stdev=0.,
	        init_logstdev=0.,
	        enable_obsnorm=enable_obsnorm)
	    policy = rl.GaussianPolicy(policy_cfg, mdp.obs_space, mdp.action_space, 'GaussianPolicy')
	else:
	    policy_cfg = rl.GibbsPolicyConfig(
	        hidden_spec=train_args['policy_hidden_spec'],
	        enable_obsnorm=enable_obsnorm)
	    policy = rl.GibbsPolicy(policy_cfg, mdp.obs_space, mdp.action_space, 'GibbsPolicy')

	policy.load_h5(policy_file, policy_key)
	
	# Generate the actions according to the expert's policy for the observations in the underperforming trajs

	expert_actions = policy.sample_actions(obs.reshape((-1,obs.shape[-1])))[0].reshape((-1,a.shape[1],a.shape[2]))
	

	# Calcualating the deviation histogram:
	action_deviations = np.linalg.norm(expert_actions.reshape((-1,a.shape[-1])) - a.reshape((-1,a.shape[-1])), axis=1)
	if generate_plot:
		plt.figure()
		plt.hist(action_deviations, bins=100)
		plt.savefig('deviation_of_agent_actions_from_expert_actions_for_observations_from_underperforming_learner_trajectories.png')
		plt.show()	
	if ipython_after_eval:
		import IPython; IPython.embed() 


def find_overperforming_trajectories(trajectories, upper_bound_reward, use_percent=False):
	obs_B_T_Do, a_B_T_Da, r_B_T, len_B = load_dataset(trajectories)
	print(type(obs_B_T_Do))
	traj_rewards =r_B_T.sum(axis=1)
	# if generate_plot:
	# 	plt.figure()
	# 	plt.hist(traj_rewards, bins=100)
	# 	plt.savefig(trajectories[:-3]+'_traj_reward_histogram.png')
	# 	plt.show()

	# if ipython_after_eval:
	# 	import IPython; IPython.embed()    
	if use_percent:
		upper_bound_reward = np.percentile(traj_rewards,upper_bound_reward)
	print "upper_bound_reward: ", upper_bound_reward

	return obs_B_T_Do[traj_rewards>upper_bound_reward], a_B_T_Da[traj_rewards>upper_bound_reward], r_B_T[traj_rewards>upper_bound_reward], len_B[traj_rewards>upper_bound_reward]


def find_underperforming_trajectories(trajectories, lower_bound_reward, use_percent=False):
	obs_B_T_Do, a_B_T_Da, r_B_T, len_B = load_dataset(trajectories)
	print(type(obs_B_T_Do))
	traj_rewards =r_B_T.sum(axis=1)
	# if generate_plot:
	# 	plt.figure()
	# 	plt.hist(traj_rewards, bins=100)
	# 	plt.savefig(trajectories[:-3]+'_traj_reward_histogram.png')
	# 	plt.show()

	# if ipython_after_eval:
	# 	import IPython; IPython.embed()    
	if use_percent:
		lower_bound_reward = np.percentile(traj_rewards,lower_bound_reward)

	return obs_B_T_Do[traj_rewards<lower_bound_reward], a_B_T_Da[traj_rewards<lower_bound_reward], r_B_T[traj_rewards<lower_bound_reward], len_B[traj_rewards<lower_bound_reward]

def find_trajectories_in_a_performance_interval(trajectories, lower_bound_reward, upper_bound_reward, use_percent=False):
	obs_B_T_Do, a_B_T_Da, r_B_T, len_B = load_dataset(trajectories)
	print(type(obs_B_T_Do))
	traj_rewards =r_B_T.sum(axis=1)
	if use_percent:
		lower_bound_reward = np.percentile(traj_rewards,lower_bound_reward)
		upper_bound_reward = np.percentile(traj_rewards,upper_bound_reward)
	target_indices = np.logical_and(traj_rewards>=lower_bound_reward, traj_rewards<=upper_bound_reward)
	return obs_B_T_Do[target_indices], a_B_T_Da[target_indices], r_B_T[target_indices], len_B[target_indices]

def find_trajectories_in_a_length_interval(trajectories, lower_bound_length, upper_bound_length, use_percent=False):
	obs_B_T_Do, a_B_T_Da, r_B_T, len_B = load_dataset(trajectories)
	print(type(obs_B_T_Do))
	# traj_rewards =r_B_T.sum(axis=1)
	if use_percent:
		lower_bound_length = np.percentile(len_B,lower_bound_length)
		upper_bound_length = np.percentile(len_B,upper_bound_length)
	target_indices = np.logical_and(len_B>=lower_bound_length, len_B<=upper_bound_length)
	return obs_B_T_Do[target_indices], a_B_T_Da[target_indices], r_B_T[target_indices], len_B[target_indices]



def tsne_analysis_of_state_space(trajectories, n_samples):
	obs_B_T_Do, a_B_T_Da, r_B_T, len_B = load_dataset(trajectories)
	obs_flattened_BT_Do = obs_B_T_Do.reshape((-1,obs_B_T_Do.shape[-1]))
	model = TSNE()
	tsne_transformed_obs_BT_2 = model.fit_transform(obs_flattened_BT_Do[np.random.randint(low=0,high=obs_flattened_BT_Do.shape[0],size=n_samples)])
	plt.figure()
	plt.scatter(tsne_transformed_obs_BT_2[:,0], tsne_transformed_obs_BT_2[:,1])
	plt.savefig('tsne_plot_of_states_from_expert_trajectories.png')
	plt.show()

def tsne_plot_states_from_good_and_bad_trajs(trajectories, lower_bound_reward, upper_bound_reward, n_samples):
	bad_obs_B_T_Do, bad_a_B_T_Da, bad_r_B_T, bad_len_B = find_underperforming_trajectories(trajectories, lower_bound_reward)
	good_obs_B_T_Do, good_a_B_T_Da, good_r_B_T, good_len_B = find_overperforming_trajectories(trajectories, upper_bound_reward)
	bad_obs_flattened = bad_obs_B_T_Do.reshape((-1,bad_obs_B_T_Do.shape[-1]))
	good_obs_flattened = good_obs_B_T_Do.reshape((-1,good_obs_B_T_Do.shape[-1]))
	data = np.concatenate((bad_obs_flattened[np.random.randint(low=0,high=bad_obs_flattened.shape[0],size=n_samples)],
		good_obs_flattened[np.random.randint(low=0,high=good_obs_flattened.shape[0],size=n_samples)]),
		axis=0)
	# model1 = TSNE()
	# data_2D = model1.fit_transform(data)


	# obs_B_T_Do, a_B_T_Da, r_B_T, len_B = load_dataset(trajectories)
	# obs_flattened_BT_Do = obs_B_T_Do.reshape((-1,obs_B_T_Do.shape[-1]))	

	# data = np.concatenate((data, obs_flattened_BT_Do[np.random.randint(low=0,high=obs_flattened_BT_Do.shape[0],size=5*n_samples)]),axis=0)
	# model2 = TSNE()
	# tsne_transformed_obs_BT_2 = model2.fit_transform(obs_flattened_BT_Do)
	model1 = TSNE()
	data_2D = model1.fit_transform(data)

	plt.figure()
	# plt.scatter(data_2D[2*n_samples:,0], data_2D[2*n_samples:,1], marker='o',alpha=0.25, facecolors='b')
	plt.scatter(data_2D[:n_samples,0], data_2D[:n_samples,1], marker='v', facecolors="none", edgecolors='r', linewidth='3')
	plt.scatter(data_2D[n_samples:2*n_samples,0], data_2D[n_samples:2*n_samples,1], marker='^', facecolors="none", edgecolors='g', linewidth='1')

	# plt.scatter(tsne_transformed_obs_BT_2[:,0], tsne_transformed_obs_BT_2[:,1], marker='o')
	plt.savefig('tsne_plot_of_states_from_good_and_bad_trajectories.png')
	plt.show()

def tsne_plot_states_from_good_and_bad_trajs_wrt_learner_states(trajectories, lower_bound_reward, upper_bound_reward, n_samples):
	bad_obs_B_T_Do, bad_a_B_T_Da, bad_r_B_T, bad_len_B = find_underperforming_trajectories(trajectories, lower_bound_reward)
	good_obs_B_T_Do, good_a_B_T_Da, good_r_B_T, good_len_B = find_overperforming_trajectories(trajectories, upper_bound_reward)
	bad_obs_flattened = bad_obs_B_T_Do.reshape((-1,bad_obs_B_T_Do.shape[-1]))
	good_obs_flattened = good_obs_B_T_Do.reshape((-1,good_obs_B_T_Do.shape[-1]))
	data = np.concatenate((bad_obs_flattened[np.random.randint(low=0,high=bad_obs_flattened.shape[0],size=n_samples)],
		good_obs_flattened[np.random.randint(low=0,high=good_obs_flattened.shape[0],size=n_samples)]),
		axis=0)
	# model1 = TSNE()
	# data_2D = model1.fit_transform(data)


	obs_B_T_Do, a_B_T_Da, r_B_T, len_B = load_dataset(trajectories)
	obs_flattened_BT_Do = obs_B_T_Do.reshape((-1,obs_B_T_Do.shape[-1]))	

	data = np.concatenate((data, obs_flattened_BT_Do[np.random.randint(low=0,high=obs_flattened_BT_Do.shape[0],size=5*n_samples)]),axis=0)
	# model2 = TSNE()
	# tsne_transformed_obs_BT_2 = model2.fit_transform(obs_flattened_BT_Do)
	model1 = TSNE()
	data_2D = model1.fit_transform(data)

	plt.figure()
	plt.scatter(data_2D[2*n_samples:,0], data_2D[2*n_samples:,1], marker='o',alpha=0.25, facecolors='b')
	plt.scatter(data_2D[:n_samples,0], data_2D[:n_samples,1], marker='v', facecolors="none", edgecolors='r', linewidth='1')
	plt.scatter(data_2D[n_samples:2*n_samples,0], data_2D[n_samples:2*n_samples,1], marker='^', facecolors="none", edgecolors='g', linewidth='1')

	# plt.scatter(tsne_transformed_obs_BT_2[:,0], tsne_transformed_obs_BT_2[:,1], marker='o')
	plt.savefig('tsne_plot_of_states_from_good_and_bad_trajectories_wrt_learner_states.png')
	plt.show()

def tsne_plot_states_from_good_and_bad_trajs_wrt_expert_states(learner_trajectories, expert_trajectories, lower_bound_reward, upper_bound_reward, n_samples, length_filtering_expert_trajs):
	bad_obs_B_T_Do, bad_a_B_T_Da, bad_r_B_T, bad_len_B = find_underperforming_trajectories(learner_trajectories, lower_bound_reward)
	good_obs_B_T_Do, good_a_B_T_Da, good_r_B_T, good_len_B = find_overperforming_trajectories(learner_trajectories, upper_bound_reward)
	bad_obs_flattened = bad_obs_B_T_Do.reshape((-1,bad_obs_B_T_Do.shape[-1]))
	good_obs_flattened = good_obs_B_T_Do.reshape((-1,good_obs_B_T_Do.shape[-1]))
	data = np.concatenate((bad_obs_flattened[np.random.randint(low=0,high=bad_obs_flattened.shape[0],size=n_samples)],
		good_obs_flattened[np.random.randint(low=0,high=good_obs_flattened.shape[0],size=n_samples)]),
		axis=0)
	# model1 = TSNE()
	# data_2D = model1.fit_transform(data)


	obs_B_T_Do, a_B_T_Da, r_B_T, len_B = load_dataset(expert_trajectories, len_filtering=length_filtering_expert_trajs)
	obs_flattened_BT_Do = obs_B_T_Do.reshape((-1,obs_B_T_Do.shape[-1]))	

	data = np.concatenate((data, obs_flattened_BT_Do[np.random.randint(low=0,high=obs_flattened_BT_Do.shape[0],size=5*n_samples)]),axis=0)
	# model2 = TSNE()
	# tsne_transformed_obs_BT_2 = model2.fit_transform(obs_flattened_BT_Do)
	model1 = TSNE()
	data_2D = model1.fit_transform(data)

	plt.figure()
	plt.scatter(data_2D[2*n_samples:,0], data_2D[2*n_samples:,1], marker='o', facecolors='b')
	plt.scatter(data_2D[:n_samples,0], data_2D[:n_samples,1], marker='v', facecolors="none", edgecolors='r', linewidth='1')
	plt.scatter(data_2D[n_samples:2*n_samples,0], data_2D[n_samples:2*n_samples,1], marker='^', facecolors="none", edgecolors='g', linewidth='1')

	# plt.scatter(tsne_transformed_obs_BT_2[:,0], tsne_transformed_obs_BT_2[:,1], marker='o')
	plt.savefig('tsne_plot_of_states_from_good_and_bad_trajectories_wrt_expert_states.png')
	plt.show()
	plt.figure()
	plt.scatter(data_2D[2*n_samples:,0], data_2D[2*n_samples:,1], marker='o',alpha=0.25, facecolors='b')
	plt.savefig('tsne_plot_of_states_from_expert.png')
	plt.show()

def find_correlation_dc_expert_traj_reward(learner_trajectories, expert_trajectories, learner_lower_bound_reward, expert_bound_percent, expert_bound_is_lower, n_samples):
	bad_obs_B_T_Do, bad_a_B_T_Da, bad_r_B_T, bad_len_B = find_underperforming_trajectories(learner_trajectories, learner_lower_bound_reward)
	if expert_bound_is_lower:
		ex_obs_B_T_Do, ex_a_B_T_Da, ex_r_B_T, ex_len_B = find_underperforming_trajectories(expert_trajectories, expert_bound_percent, use_percent=True)
	else:
		ex_obs_B_T_Do, ex_a_B_T_Da, ex_r_B_T, ex_len_B = find_overperforming_trajectories(expert_trajectories, expert_bound_percent, use_percent=True)
	
	bad_obs_flattened = bad_obs_B_T_Do.reshape((-1,bad_obs_B_T_Do.shape[-1]))
	ex_obs_flattened = ex_obs_B_T_Do.reshape((-1,ex_obs_B_T_Do.shape[-1]))

	data = np.concatenate((bad_obs_flattened[np.random.randint(low=0,high=bad_obs_flattened.shape[0],size=n_samples)],
		ex_obs_flattened[np.random.randint(low=0,high=ex_obs_flattened.shape[0],size=n_samples)]),
		axis=0)	
	model1 = TSNE()
	data_2D = model1.fit_transform(data)

	plt.figure()
	# plt.scatter(data_2D[2*n_samples:,0], data_2D[2*n_samples:,1], marker='o', facecolors='b')
	l = plt.scatter(data_2D[:n_samples,0], data_2D[:n_samples,1], marker='v', facecolors="none", edgecolors='r', linewidth='1')
	e = plt.scatter(data_2D[n_samples:,0], data_2D[n_samples:,1], marker='^', facecolors="none", edgecolors='g', linewidth='1')

	plt.legend((l,e),
		('learner','expert'),
		scatterpoints=1)
	if expert_bound_is_lower:
		f_name = 'tsne_plot_of_states_from_bad_learner_trajectories_wrt_lower_%0.1f_expert_states.png' % expert_bound_percent
	else:
		f_name = 'tsne_plot_of_states_from_bad_learner_trajectories_wrt_upper_%0.1f_expert_states.png' % expert_bound_percent

	plt.savefig(f_name)
	plt.show()

def find_correlation_dc_expert_traj_reward_range(learner_trajectories, expert_trajectories, learner_lower_bound_reward, expert_lower_bound_percent, expert_upper_bound_percent, n_samples):
	bad_obs_B_T_Do, bad_a_B_T_Da, bad_r_B_T, bad_len_B = find_underperforming_trajectories(learner_trajectories, learner_lower_bound_reward)
	
	ex_obs_B_T_Do, ex_a_B_T_Da, ex_r_B_T, ex_len_B = find_trajectories_in_a_performance_interval(expert_trajectories, expert_lower_bound_percent, expert_upper_bound_percent, use_percent=True)
	
	bad_obs_flattened = bad_obs_B_T_Do.reshape((-1,bad_obs_B_T_Do.shape[-1]))
	ex_obs_flattened = ex_obs_B_T_Do.reshape((-1,ex_obs_B_T_Do.shape[-1]))

	data = np.concatenate((bad_obs_flattened[np.random.randint(low=0,high=bad_obs_flattened.shape[0],size=n_samples)],
		ex_obs_flattened[np.random.randint(low=0,high=ex_obs_flattened.shape[0],size=n_samples)]),
		axis=0)	
	model1 = TSNE()
	data_2D = model1.fit_transform(data)

	plt.figure()
	# plt.scatter(data_2D[2*n_samples:,0], data_2D[2*n_samples:,1], marker='o', facecolors='b')
	l = plt.scatter(data_2D[:n_samples,0], data_2D[:n_samples,1], marker='v', facecolors="none", edgecolors='r', linewidth='1')
	e = plt.scatter(data_2D[n_samples:,0], data_2D[n_samples:,1], marker='^', facecolors="none", edgecolors='g', linewidth='1')

	plt.legend((l,e),
		('learner','expert'),
		scatterpoints=1)

	f_name = 'tsne_plot_of_states_from_bad_learner_trajectories_wrt_expert_states_in_%0.1f_to_%0.1f_percentile_trajectories.png' % (expert_lower_bound_percent, expert_upper_bound_percent)

	plt.savefig(f_name)
	# plt.show()

def find_correlation_dc_expert_traj_length_range(learner_trajectories, expert_trajectories, learner_lower_bound_reward, expert_lower_bound_percent, expert_upper_bound_percent, n_samples):
	bad_obs_B_T_Do, bad_a_B_T_Da, bad_r_B_T, bad_len_B = find_underperforming_trajectories(learner_trajectories, learner_lower_bound_reward)
	
	ex_obs_B_T_Do, ex_a_B_T_Da, ex_r_B_T, ex_len_B = find_trajectories_in_a_length_interval(expert_trajectories, expert_lower_bound_percent, expert_upper_bound_percent, use_percent=True)
	
	bad_obs_flattened = bad_obs_B_T_Do.reshape((-1,bad_obs_B_T_Do.shape[-1]))
	ex_obs_flattened = ex_obs_B_T_Do.reshape((-1,ex_obs_B_T_Do.shape[-1]))

	data = np.concatenate((bad_obs_flattened[np.random.randint(low=0,high=bad_obs_flattened.shape[0],size=n_samples)],
		ex_obs_flattened[np.random.randint(low=0,high=ex_obs_flattened.shape[0],size=n_samples)]),
		axis=0)	
	model1 = TSNE()
	data_2D = model1.fit_transform(data)

	plt.figure()
	# plt.scatter(data_2D[2*n_samples:,0], data_2D[2*n_samples:,1], marker='o', facecolors='b')
	l = plt.scatter(data_2D[:n_samples,0], data_2D[:n_samples,1], marker='v', facecolors="none", edgecolors='r', linewidth='1')
	e = plt.scatter(data_2D[n_samples:,0], data_2D[n_samples:,1], marker='^', facecolors="none", edgecolors='g', linewidth='1')

	plt.legend((l,e),
		('learner','expert'),
		scatterpoints=1)

	f_name = 'length_tsne_plot_of_states_from_bad_learner_trajectories_wrt_expert_states_in_%0.1f_to_%0.1f_percentile_trajectories.png' % (expert_lower_bound_percent, expert_upper_bound_percent)

	plt.savefig(f_name)
	# plt.show()

def create_length_histogram(trajectories):
	_,_,_,len_B = load_dataset(trajectories)
	plt.figure()
	plt.hist(len_B,bins=100)
	plt.savefig('traj_length_histogram.png')
	plt.show()

def correlate_reward_to_length_of_trajectory(trajectories):
	obs_B_T_Do, a_B_T_Da, r_B_T, len_B = load_dataset(trajectories)
	traj_rewards = r_B_T.sum(axis=1)
	data = []
	plt.figure()
	for i in range(10):
		if i==0:
			for j in range(10):			
				lower_bound_reward = np.percentile(traj_rewards, j)
				upper_bound_reward = np.percentile(traj_rewards, j+1.)
				indices = np.logical_and(traj_rewards>=lower_bound_reward, traj_rewards<=upper_bound_reward)
				filtered_lengths = len_B[indices]
				data.append([(lower_bound_reward+upper_bound_reward)/2. , filtered_lengths.mean()])
		else:
			lower_bound_reward = np.percentile(traj_rewards, i*10.)
			upper_bound_reward = np.percentile(traj_rewards, j*10.+10.)
			indices = np.logical_and(traj_rewards>=lower_bound_reward, traj_rewards<=upper_bound_reward)
			filtered_lengths = len_B[indices]
			data.append([(lower_bound_reward+upper_bound_reward)/2. , filtered_lengths.mean()])

		# print "%0.1f to %0.1f: "%(lower_bound_reward,upper_bound_reward), filtered_lengths.mean(), "count: ", len(filtered_lengths)
	data = np.asarray(data)
	plt.plot(data[:,0], data[:,1], marker='o')
	plt.xlabel('reward')
	plt.ylabel('length')
	plt.savefig("lengths_of_different_reward_intervals.png")
	plt.show()

	# 	plt.subplot(10,1,i+1)
	# 	plt.hist(filtered_lengths)
	# plt.savefig("lengths_of_different_reward_intervals.png")
	# plt.show()

def plot_reward_histogram(trajectories):
	obs_B_T_Do, a_B_T_Da, r_B_T, len_B = load_dataset(trajectories)
	traj_rewards = r_B_T.sum(axis=1)
	plt.figure()
	# plt.hist(traj_rewards,bins=100)
	plt.hist(traj_rewards,bins=[i for i in xrange(0,3600,40)])
	plt.title("HalfCheetah - Histogram of rewards of the VanillaGAIL-agent's trajectories")
	plt.xlabel("Total trajectory reward")
	plt.ylabel("Number of trajectories")
	# plt.savefig("histogram_of_trajectory_rewards_"+trajectories.split('/')[-1][:-3]+".png")
	plt.savefig("histogram_of_trajectory_rewards_"+trajectories.split('/')[-1]+".png")
	plt.show()

def print_mean_std(trajectories):
	obs_B_T_Do, a_B_T_Da, r_B_T, len_B = load_dataset(trajectories)
	traj_rewards = r_B_T.sum(axis=1)
	print "Trajectory reward mean: %f, std: %f" %(traj_rewards.mean(),traj_rewards.std())


def embed_and_plot_good_vs_bad_trajs(trajectories, n_dims_tsne):
	"""
	Under development
	"""
	obs_B_T_Do, a_B_T_Da, r_B_T, len_B = load_dataset(trajectories)
	traj_rewards = r_B_T.sum(axis=1)
	lower_percentile = np.percentile(traj_rewards,10)
	upper_percentile = np.percentile(traj_rewards,90)
	bad_traj_indices = traj_rewards<=lower_percentile
	good_traj_indices = traj_rewards>=upper_percentile
	bad_obs_B_T_Do, bad_a_B_T_Da, bad_r_B_T, bad_len_B = obs_B_T_Do[bad_traj_indices], a_B_T_Da[bad_traj_indices], r_B_T[bad_traj_indices], len_B[bad_traj_indices]
	good_obs_B_T_Do, good_a_B_T_Da, good_r_B_T, good_len_B = obs_B_T_Do[good_traj_indices], a_B_T_Da[good_traj_indices], r_B_T[good_traj_indices], len_B[good_traj_indices] 
	
	all_obs_alldims = np.concatenate((bad_obs_B_T_Do.reshape((-1,bad_obs_B_T_Do.shape[-1])),
			good_obs_B_T_Do.reshape((-1,good_obs_B_T_Do.shape[-1]))),
		axis=0)
	all_a_alldims = np.concatenate((bad_a_B_T_Da.reshape((-1,bad_a_B_T_Da.shape[-1])),
		good_a_B_T_Da.reshape((-1,good_a_B_T_Da.shape[-1]))),
		axis=0)
	all_len = np.concatenate((bad_len_B,good_len_B),axis=0)
	all_len_cumsum = np.cumsum(all_len)

	model_obs = TSNE()
	all_obs_1dim = model_obs.fit_transform(all_obs_alldims)
	model_a = TSNE()
	all_a_1dim = model_a.fit_transform(all_a_alldims)

	all_obs_1dim_unstacked = np.split(all_obs_1dim,all_len_cumsum[:-1])
	all_a_1dim_unstacked = np.split(all_a_1dim,all_len_cumsum[:-1])

	bad_obs_1dim = all_obs_1dim_unstacked[:bad_len_B.shape[0]]
	good_obs_1dim = all_obs_1dim_unstacked[bad_len_B.shape[0]:]

	# print 
def estimate_state_density(trajectories, n_components, length_filtering_expert_trajs):	
	obs_B_T_Do, a_B_T_Da, r_B_T, len_B = load_dataset(trajectories, len_filtering=length_filtering_expert_trajs)
	obs_flattened = obs_B_T_Do.reshape((-1,obs_B_T_Do.shape[-1]))
	model = GMM(n_components=n_components, covariance_type='diag')
	model.fit(obs_flattened)
	return model 


def tsne_plot_states_from_good_and_bad_trajs_wrt_expert_states_with_density_estimation(learner_trajectories, expert_trajectories, lower_bound_reward, upper_bound_reward, n_samples, length_filtering_expert_trajs):
	"""Under development"""
	bad_obs_B_T_Do, bad_a_B_T_Da, bad_r_B_T, bad_len_B = find_underperforming_trajectories(learner_trajectories, lower_bound_reward)
	good_obs_B_T_Do, good_a_B_T_Da, good_r_B_T, good_len_B = find_overperforming_trajectories(learner_trajectories, upper_bound_reward)
	bad_obs_flattened = bad_obs_B_T_Do.reshape((-1,bad_obs_B_T_Do.shape[-1]))
	good_obs_flattened = good_obs_B_T_Do.reshape((-1,good_obs_B_T_Do.shape[-1]))
	data = np.concatenate((bad_obs_flattened[np.random.randint(low=0,high=bad_obs_flattened.shape[0],size=n_samples)],
		good_obs_flattened[np.random.randint(low=0,high=good_obs_flattened.shape[0],size=n_samples)]),
		axis=0)
	# model1 = TSNE()
	# data_2D = model1.fit_transform(data)


	obs_B_T_Do, a_B_T_Da, r_B_T, len_B = load_dataset(expert_trajectories, len_filtering=length_filtering_expert_trajs)
	obs_flattened_BT_Do = obs_B_T_Do.reshape((-1,obs_B_T_Do.shape[-1]))	

	data = np.concatenate((data, obs_flattened_BT_Do[np.random.randint(low=0,high=obs_flattened_BT_Do.shape[0],size=5*n_samples)]),axis=0)
	# model2 = TSNE()
	# tsne_transformed_obs_BT_2 = model2.fit_transform(obs_flattened_BT_Do)
	model1 = TSNE()
	data_2D = model1.fit_transform(data)

	#Train a GMM density model
	t0 = time()
	gmm = estimate_state_density(expert_trajectories, n_components=1, length_filtering_expert_trajs=length_filtering_expert_trajs)
	t1 = time()
	scores_bad = gmm.score_samples(bad_obs_flattened)
	scores_good = gmm.score_samples(good_obs_flattened)
	t2 = time()
	print "Number of GMM components = 1"
	print "Mean score of good states: %f, bad states: %f" %(scores_good.mean(), scores_bad.mean())
	print "Std score of good states: %f, bad states: %f" %(scores_good.std(), scores_bad.std())
	print "Min score of good states: %f, bad states: %f" %(scores_good.min(), scores_bad.min())
	print "Max score of good states: %f, bad states: %f" %(scores_good.max(), scores_bad.max())
	print "Time to fit GMM: %f, inference rate: %f per sample" %(t1-t0, (t2-t1)/(scores_bad.shape[0]+scores_good.shape[0]))

	plt.figure()
	plt.hist(scores_good,bins=10)
	plt.savefig('good_state_gmm_scores.png')
	plt.show()
	plt.figure()
	plt.hist(scores_bad,bins=10)
	plt.savefig('bad_state_gmm_scores.png')
	plt.show()


	# plt.figure()
	# plt.scatter(data_2D[2*n_samples:,0], data_2D[2*n_samples:,1], marker='o', facecolors='b')
	# plt.scatter(data_2D[:n_samples,0], data_2D[:n_samples,1], marker='v', facecolors="none", edgecolors='r', linewidth='1')
	# plt.scatter(data_2D[n_samples:2*n_samples,0], data_2D[n_samples:2*n_samples,1], marker='^', facecolors="none", edgecolors='g', linewidth='1')

	# # plt.scatter(tsne_transformed_obs_BT_2[:,0], tsne_transformed_obs_BT_2[:,1], marker='o')
	# plt.savefig('tsne_plot_of_states_from_good_and_bad_trajectories_wrt_expert_states.png')
	# plt.show()
	# plt.figure()
	# plt.scatter(data_2D[2*n_samples:,0], data_2D[2*n_samples:,1], marker='o',alpha=0.25, facecolors='b')
	# plt.savefig('tsne_plot_of_states_from_expert.png')
	# plt.show()


def fit_gmm_and_evaluate_scores(trajectory_set1, trajectory_set2, length_filtering_expert_trajs, upper_percentile, lower_percentile, n_components_gmm):
	gmm = estimate_state_density(trajectory_set1, n_components_gmm, length_filtering_expert_trajs)
	bad_obs_B_T_Do, bad_a_B_T_Da, bad_r_B_T, bad_len_B = find_underperforming_trajectories(trajectory_set2, lower_percentile, use_percent=True)
	good_obs_B_T_Do, good_a_B_T_Da, good_r_B_T, good_len_B = find_overperforming_trajectories(trajectory_set2, upper_percentile, use_percent=True)
	bad_obs_flattened = bad_obs_B_T_Do.reshape((-1,bad_obs_B_T_Do.shape[-1]))
	good_obs_flattened = good_obs_B_T_Do.reshape((-1,good_obs_B_T_Do.shape[-1]))	
	scores_bad = gmm.score_samples(bad_obs_flattened)
	scores_good = gmm.score_samples(good_obs_flattened)	
	print "Mean score of good states: %f, bad states: %f" %(scores_good.mean(), scores_bad.mean())
	print "Std score of good states: %f, bad states: %f" %(scores_good.std(), scores_bad.std())
	print "Min score of good states: %f, bad states: %f" %(scores_good.min(), scores_bad.min())
	print "Max score of good states: %f, bad states: %f" %(scores_good.max(), scores_bad.max())	
	print "Bad percentiles:"
	for i in range(11):
		p = np.percentile(scores_bad,i*10.)
		print "%d percentile: %f" % (i*10, p)
	print "Good percentiles:"
	for i in range(11):
		p = np.percentile(scores_good,i*10.)
		print "%d percentile: %f" % (i*10, p)


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--reference_trajectories', type=str, required=True, help="Path to .h5 file containing reference trajectories")
	parser.add_argument('--trajectories_to_be_compared', type=str, required=False, help="Path to .h5 file containing trajectories to be compared")
	parser.add_argument('--policy_to_be_compared', type=str, help="Path to .h5 file containing learner policy")
	parser.add_argument('--env_name', type=str, required=True)
	parser.add_argument('--limit_trajs', type=int, help="How many expert trajectories to be used for training. If None : full dataset is used.")
	parser.add_argument('--data_subsamp_freq', type=int, help="A number between 0 and max_traj_len. Rate of subsampling of expert trajectories while creating the dataset of expert transitions (state-action)")
	parser.add_argument('--policy_hidden_spec', type=str, default=SIMPLE_ARCHITECTURE)
	parser.add_argument('--ipython_after_eval', action='store_true')
	parser.add_argument('--max_traj_len', type=int, default=None) # only used for saving

	parser.add_argument('--lower_bound_reward', type=float, default=1000)
	parser.add_argument('--upper_bound_reward', type=float, default=9000)
	parser.add_argument('--generate_plot', action='store_true')
	parser.add_argument('--num_samples_tsne',type=int, default=1000)
	parser.add_argument('--expert_bound_lower', type=int, default=10)
	parser.add_argument('--expert_bound_upper', type=int, default=90)

	parser.add_argument('--length_filtering_expert_trajs', action='store_true')
	# parser.add_argument('--expert_bound_is_lower', action='store_true')

	args = parser.parse_args()

	# find_deviation_of_agent_actions_from_expert_actions_for_observations_from_expert_trajectories(args.reference_trajectories, args.policy_to_be_compared, args.limit_trajs, args.data_subsamp_freq, args.ipython_after_eval)
	# obs,a,r,l = find_underperforming_trajectories(args.reference_trajectories, args.policy_to_be_compared, args.lower_bound_reward, args.ipython_after_eval, args.generate_plot)
	# find_deviation_of_agent_actions_from_expert_actions_for_underperforming_trajectories(args.reference_trajectories, args.policy_to_be_compared, args.lower_bound_reward, args.ipython_after_eval, args.generate_plot)
	# tsne_analysis_of_state_space(args.reference_trajectories, args.num_samples_tsne)
	# tsne_plot_states_from_good_and_bad_trajs_wrt_learner_states(args.reference_trajectories, args.lower_bound_reward, args.upper_bound_reward, args.num_samples_tsne)
	# tsne_plot_states_from_good_and_bad_trajs_wrt_expert_states_with_density_estimation(args.reference_trajectories, args.trajectories_to_be_compared, args.lower_bound_reward, args.upper_bound_reward, args.num_samples_tsne, args.length_filtering_expert_trajs)
	# tsne_plot_states_from_good_and_bad_trajs(args.reference_trajectories, args.lower_bound_reward, args.upper_bound_reward, args.num_samples_tsne)
	# find_correlation_dc_expert_traj_reward(args.reference_trajectories, args.trajectories_to_be_compared, args.lower_bound_reward, args.expert_bound, args.expert_bound_is_lower, args.num_samples_tsne)
	# find_correlation_dc_expert_traj_reward_range(args.reference_trajectories, args.trajectories_to_be_compared, args.lower_bound_reward, args.expert_bound_lower, args.expert_bound_upper, args.num_samples_tsne)
	# for i in range(10):
	# 	find_correlation_dc_expert_traj_length_range(args.reference_trajectories, args.trajectories_to_be_compared, args.lower_bound_reward, i*10., i*10.+10., args.num_samples_tsne)
	# create_length_histogram(args.trajectories_to_be_compared)
	# correlate_reward_to_length_of_trajectory(args.trajectories_to_be_compared)
	plot_reward_histogram(args.reference_trajectories)
	# print_mean_std(args.reference_trajectories)
	# fit_gmm_and_evaluate_scores(args.reference_trajectories, args.trajectories_to_be_compared, args.length_filtering_expert_trajs, args.expert_bound_upper, args.expert_bound_lower, 1)