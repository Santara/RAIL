from environments import rlgymenv
import policyopt
from policyopt import SimConfig, rl, util, nn, tqdm
import gym
import numpy as np 
import argparse

def main():
	np.set_printoptions(suppress=True, precision=5, linewidth=1000)

	parser = argparse.ArgumentParser()
	parser.add_argument('env', type=str)
	parser.add_argument('--num_eval_trajs', type=int, default=50)
	parser.add_argument('--max_traj_len', type=int, default=None)
	parser.add_argument('--out', type=str, default=None)

	args=parser.parse_args()

	# Initialize the mdp
	mdp = rlgymenv.RLGymMDP(args.env)
	env = gym.make(args.env)
	print "Initialized environment %s" % args.env
	util.header('MDP observation space, action space sizes: %d, %d\n' % (mdp.obs_space.dim, mdp.action_space.storage_size))

	if args.max_traj_len is None:
	    	args.max_traj_len = mdp.env_spec.timestep_limit
	util.header('Max traj len is {}'.format(args.max_traj_len))

	# Run the simulation
	returns = []
	lengths = []
	sim = mdp.new_sim()

	for i_traj in range(args.num_eval_trajs):
		print i_traj, args.num_eval_trajs
		sim.reset()
		totalr = 0.
		l = 0
		while not sim.done and l < args.max_traj_len:
			#a = [np.random.uniform(mdp.action_space.low[i], mdp.action_space.high[i]) for i in range(len(mdp.action_space.shape[0]))]
			a = env.action_space.sample()
			if isinstance(mdp.action_space, policyopt.FiniteSpace):
				a = np.asarray([a])
			r = sim.step(a)
			totalr += r 
			l += 1
		returns.append(totalr)
		lengths.append(l)
	print "Mean reward: {}, Std reward: {}, Mean length: {}, Std length: {}\n".format(np.asarray(returns).mean(), np.asarray(returns).std(), np.asarray(lengths).mean(), np.asarray(lengths).std())
	if args.out is not None:
		with open(args.out,'w') as f:
			f.write("Mean reward: {}, Std reward: {}, Mean length: {}, Std length: {}\n".format(np.asarray(returns).mean(), np.asarray(returns).std(), np.asarray(lengths).mean(), np.asarray(lengths).std()))
			f.close()

if __name__=='__main__':
	main()

