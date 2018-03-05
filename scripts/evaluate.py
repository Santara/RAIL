import argparse 
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib.ticker import FuncFormatter
import h5py
from IPython.core.debugger import Pdb
from sklearn.mixture import GaussianMixture as GMM
import math

EXPERT_MEAN_Humanoid = 9790.99989735
EXPERT_STD_Humanoid = 1175.06649136

EXPERT_MEAN_Hopper = 3666.88
EXPERT_STD_Hopper = 392.908

EXPERT_MEAN_Walker = 5317.64
EXPERT_STD_Walker = 655.55

EXPERT_MEAN_HalfCheetah = 3583.946
EXPERT_STD_HalfCheetah = 101.54

EXPERT_MEAN_MountainCar = -99.8		# GAIL paper : -98.75
EXPERT_STD_MountainCar = 6.6		# GAIL paper : 8.71

EXPERT_MEAN_MountainCar = -4.08		# GAIL paper : -4.09
EXPERT_STD_MountainCar = 1.44		# GAIL paper : 1.70

numTrajs = 0

def load_dataset(filename):
	# Load expert data
	with h5py.File(filename, 'r') as f:
		# Read data as written by vis_mj.py
		dset_size = full_dset_size = f['obs_B_T_Do'].shape[0] # full dataset size
		# dset_size = min(full_dset_size, limit_trajs) if limit_trajs is not None else full_dset_size

		exobs_B_T_Do = f['obs_B_T_Do'][:dset_size,...][...]
		exa_B_T_Da = f['a_B_T_Da'][:dset_size,...][...]
		exr_B_T = f['r_B_T'][:dset_size,...][...] #Total reward obtained from each trajectory
		exlen_B = f['len_B'][:dset_size,...][...] #Length of trajectories

	print filename
	print 'Dataset size: {} transitions ({} trajectories)'.format(exlen_B.sum(), len(exlen_B))
	print 'Average return:', exr_B_T.sum(axis=1).mean()
	print 'Average std:', exr_B_T.sum(axis=1).std()

	return exobs_B_T_Do, exa_B_T_Da, exr_B_T, exlen_B

def evaluate_q_percentile_traj_reward(trajectories, q):
	_,_,r_B_T,_ = load_dataset(trajectories)
	traj_rewards = r_B_T.sum(axis=1)
	ten_percentile_val = traj_rewards[np.where(traj_rewards < np.percentile(traj_rewards,q))[0][-1]]
	# Pdb().set_trace()
	print np.percentile(traj_rewards,q), q
	print "Range of trajectories : %.2f to %.2f" % (np.min(traj_rewards), np.max(traj_rewards))
	print "%.2f-percentile trajectory reward value is: %f\n" % (q,ten_percentile_val)
	return ten_percentile_val

def compare_q_percentile_traj_reward(trajectories_GAIL, trajectories_CVaR, q):
	# pdb.set_trace()
	q_percentile_val_GAIL = evaluate_q_percentile_traj_reward(trajectories_GAIL, q)
	q_percentile_val_CVaR = evaluate_q_percentile_traj_reward(trajectories_CVaR, q)
	print "%.2f-percentile trajectory reward values:" % (q)
	print 'GAIL : %.2f' % (q_percentile_val_GAIL)
	print 'CVaR : %.2f' % (q_percentile_val_CVaR)

def evaluate_q_CVaRvalue(trajectories, q):
	_,_,r_B_T,_ = load_dataset(trajectories)
	traj_rewards = r_B_T.sum(axis=1)
	traj_rewards = np.sort(traj_rewards)
	percentile_val = np.percentile(traj_rewards,(1.0-q)*100)
	q_CVaR_indices = np.where(traj_rewards < percentile_val)[0]
	print '------ ' + str(q_CVaR_indices.shape)
	q_VaR_value = traj_rewards[np.where(traj_rewards > percentile_val)[0][0]]		# this is the rightmost point in the reward-space, i.e. leftmost (VaR value) in the cost-space
	print '------ ' + str(np.where(traj_rewards < q_VaR_value)[0].shape)
	q_CVaR_value = np.sum(traj_rewards[q_CVaR_indices])/q_CVaR_indices.shape[0]

	# print np.percentile(traj_rewards,(1.0-q)*100), (1.0-q)
	print "Range of trajectories : %.2f to %.2f" % (np.min(traj_rewards), np.max(traj_rewards))
	print "%.2f-VaR value is: %f" % (q, q_VaR_value)
	print "%.2f-CVaR value is: %f\n" % (q, q_CVaR_value)
	return -q_VaR_value, -q_CVaR_value

def compare_q_CVaR(trajectories_GAIL, trajectories_CVaR, q):
	q_CVaRvalue_GAIL = evaluate_q_CVaRvalue(trajectories_GAIL, q)
	q_CVaRvalue_CVaR = evaluate_q_CVaRvalue(trajectories_CVaR, q)
	print "%.2f CVaR values:" % (q)
	print 'GAIL : %.2f' % (q_CVaRvalue_GAIL)
	print 'CVaR : %.2f' % (q_CVaRvalue_CVaR)

def compare_q_CVaR_all3(trajectories_expert, trajectories_GAIL, trajectories_CVaR, q):
	q_VaRvalue_exp, q_CVaRvalue_exp	 = evaluate_q_CVaRvalue(trajectories_expert, q)
	q_VaRvalue_GAIL, q_CVaRvalue_GAIL = evaluate_q_CVaRvalue(trajectories_GAIL, q)
	q_VaRvalue_CVaR, q_CVaRvalue_CVaR = evaluate_q_CVaRvalue(trajectories_CVaR, q)
	print "%.2f VaR, CVaR values:" % (q)
	print 'Exp \t: %.2f \t: %.2f' % (q_VaRvalue_exp, q_CVaRvalue_exp)
	print 'GAIL \t: %.2f \t: %.2f' % (q_VaRvalue_GAIL, q_CVaRvalue_GAIL)
	print 'RAIL \t: %.2f \t: %.2f' % (q_VaRvalue_CVaR, q_CVaRvalue_CVaR)

def generate_all_scores(trajectories_expert, trajectories_GAIL, trajectories_RAIL, q):
	q_VaRvalue_exp, q_CVaRvalue_exp	 = evaluate_q_CVaRvalue(trajectories_expert, q)
	q_VaRvalue_GAIL, q_CVaRvalue_GAIL = evaluate_q_CVaRvalue(trajectories_GAIL, q)
	q_VaRvalue_RAIL, q_CVaRvalue_RAIL = evaluate_q_CVaRvalue(trajectories_RAIL, q)
	print "%.2f VaR, CVaR values:" % (q)
	print 'Exp \t: %.2f \t: %.2f' % (q_VaRvalue_exp, q_CVaRvalue_exp)
	print 'GAIL \t: %.2f \t: %.2f' % (q_VaRvalue_GAIL, q_CVaRvalue_GAIL)
	print 'RAIL \t: %.2f \t: %.2f' % (q_VaRvalue_RAIL, q_CVaRvalue_RAIL)
	

	VaR_GAIL_E = 100.0*(q_VaRvalue_exp - q_VaRvalue_GAIL)/np.abs(q_VaRvalue_exp)
	CVaR_GAIL_E = 100.0*(q_CVaRvalue_exp - q_CVaRvalue_GAIL)/np.abs(q_CVaRvalue_exp)
	VaR_RAIL_E = 100.0*(q_VaRvalue_exp - q_VaRvalue_RAIL)/np.abs(q_VaRvalue_exp)
	CVaR_RAIL_E = 100.0*(q_CVaRvalue_exp - q_CVaRvalue_RAIL)/np.abs(q_CVaRvalue_exp)
	print "%.2f relative-VaR, relative-CVaR values:" % (q)
	print 'GAIL \t: %.2f \t: %.2f' % (VaR_GAIL_E, CVaR_GAIL_E)
	print 'RAIL \t: %.2f \t: %.2f' % (VaR_RAIL_E, CVaR_RAIL_E)

	GR_VaR = VaR_RAIL_E - VaR_GAIL_E
	GR_CVaR = CVaR_RAIL_E - CVaR_GAIL_E
	print "GR-VaR \t: %.2f" % GR_VaR
	print "GR-CVaR \t: %.2f" % GR_CVaR


path = 'FinalEvaluation/VaR_CVaR_analysis/'
expert_list = ['expert_HalfCheetah_1000', 'expert_Hopper_1000', 'expert_Humanoid_1000', 'expert_Reacher_1000', 'expert_Walker_1000']
vanilla_list = ['vanilla_HalfCheetah_1000', 'vanilla_Hopper_1000', 'vanilla_Humanoid_1000', 'vanilla_Reacher_1000', 'vanilla_Walker_1000_iter460']
CVaR_list = ['CVaR_HalfCheetah_1000', 'CVaR_Hopper_1000', 'CVaR_Humanoid_1000', 'CVaR_Reacher_1000', 'CVaR_Walker_1000_iter480']

def compare_VaR_CVaR(q):

	numEnvs = 5
	numTrajs = [32,64,128,256,512]

	for i in range(numEnvs):
		var_list_all, cvar_list_all = [],[]

		for numTraj in numTrajs:
		# for numTraj in [32,64]:	
			var_list, cvar_list = compare_q_CVaR_all3(path+expert_list[i]+'_'+str(numTraj), path+vanilla_list[i]+'_'+str(numTraj,), path+CVaR_list[i]+'_'+str(numTraj,), q)
			var_list_all.append(var_list)
			cvar_list_all.append(cvar_list)

		var_list_all = np.asarray(var_list_all)
		cvar_list_all = np.asarray(cvar_list_all) 

		print var_list_all.shape, cvar_list_all.shape
		envName = expert_list[i].split('_')[1]

		plt.figure(i)
		plt.plot(range(len(numTrajs)), var_list_all[:,0], color = 'r', label='expert_VaR')
		plt.plot(range(len(numTrajs)), var_list_all[:,1], color = 'g', label='vanilla_VaR')
		plt.plot(range(len(numTrajs)), var_list_all[:,2], color = 'b', label='ours_VaR')
		plt.xticks(range(len(numTrajs)), numTrajs)
		plt.legend()
		plt.title(envName)
		plt.savefig('VaR_analysis_' + expert_list[i].split('_')[1] + '.png')

		plt.figure(i+numEnvs)
		plt.plot(range(len(numTrajs)), cvar_list_all[:,0], color = 'r', label='expert_CVaR')
		plt.plot(range(len(numTrajs)), cvar_list_all[:,1], color = 'g', label='vanilla_CVaR')
		plt.plot(range(len(numTrajs)), cvar_list_all[:,2], color = 'b', label='ours_CVaR')
		plt.xticks(range(len(numTrajs)), numTrajs)
		plt.legend()
		plt.title(envName)
		plt.savefig('CVaR_analysis_' + envName + '.png')
		# plt.savefig('VaR_CVaR_analysis_' + envName + '.png')

def leftmost_bin_percent_population_size(trajectories, numBins):
	_,_,r_B_T,_ = load_dataset(trajectories)
	traj_rewards =r_B_T.sum(axis=1)
	bin_size = (np.max(traj_rewards)-np.min(traj_rewards))/(1.0*numBins)
	print bin_size
	print np.min(traj_rewards)
	pop_in_leftmost_bin = traj_rewards<=np.min(traj_rewards)+bin_size
	print pop_in_leftmost_bin.sum()	
	pop_in_leftmost_bin_percentage = 100*float(pop_in_leftmost_bin.sum()/float(pop_in_leftmost_bin.shape[0]))
	print "Leftmost bin population percentage: %0.2f%%" % (pop_in_leftmost_bin_percentage)
	return pop_in_leftmost_bin_percentage

def compare_leftmost(trajectories_GAIL, trajectories_CVaR, numBins):
	pop_in_leftmost_bin_percentage_GAIL = leftmost_bin_percent_population_size(trajectories_GAIL, numBins)
	pop_in_leftmost_bin_percentage_CVaR = leftmost_bin_percent_population_size(trajectories_CVaR, numBins)

	print "\n\nLeftmost bin population percentages:\n"
	print 'GAIL : %.2f%%' % (pop_in_leftmost_bin_percentage_GAIL)
	print 'CVaR : %.2f%%' % (pop_in_leftmost_bin_percentage_CVaR)

def moving_average(inp,window_size):
	filt = np.ones((window_size))
	filt = filt/len(filt)
	out = np.convolve(inp, filt, "same")
	return out

def compute_variance_wrt_expert_wrapper(training_log, expert_mean, expert_std, fraction):
	f = h5py.File(training_log,'r')
	log = f['log']

	mean_series = np.asarray([log[i][1] for i in range(log.shape[0])])
	std_series 	= np.asarray([log[i][3] for i in range(log.shape[0])])

	mean_smoothed 	= moving_average(mean_series, window_size=21)
	std_smoothed 	= moving_average(std_series, window_size=21)
	index = np.where(mean_smoothed > fraction*expert_mean)[0][0]
	# return std_series, index
	print '\n\n%s\n%d%% convergence index : %d' % (training_log, fraction*100, index)

	std_converged = std_series[index:] 
	avg_score 	= np.sum(std_converged)/std_converged.shape[0]

	print 'Avg_variance : %.2f' % (avg_score)
	return avg_score

def compute_variance_wrt_expert(training_log_GAIL, training_log_CVaR, expert_mean, expert_std, fraction):

	avg_variance_GAIL = compute_variance_wrt_expert_wrapper(training_log_GAIL, expert_mean, expert_std, fraction)
	avg_variance_CVaR = compute_variance_wrt_expert_wrapper(training_log_CVaR, expert_mean, expert_std, fraction)

	print '\nRatios :'
	print 'GAIL :: %.2f' % (avg_variance_GAIL/expert_std)
	print 'CVaR :: %.2f' % (avg_variance_CVaR/expert_std)

def compute_metrics(training_log, expert_mean, expert_std, fraction, reward_offset):
	f = h5py.File(training_log,'r')
	log = f['log']
	mean_series = np.asarray([log[i][1] for i in range(log.shape[0])]) + reward_offset
	std_series 	= np.asarray([log[i][3] for i in range(log.shape[0])])
	deviation 	= std_series - expert_std

	mean_smoothed 	= moving_average(mean_series, window_size=21)
	std_smoothed 	= moving_average(std_series, window_size=21)
	index = np.where(mean_smoothed > fraction*expert_mean)[0][0]
	# return std_series, index
	print '\n\n%s\n%d%% convergence index : %d' % (training_log, fraction*100, index)

	good_indicator 	= np.where(std_series[index:] < std_smoothed[index:])[0]
	bad_indicator 	= np.where(std_series[index:] > std_smoothed[index:]) [0]

	deviation_converged = deviation[index:] 
	good_score 	= np.sum(deviation_converged[good_indicator])/good_indicator.shape[0]
	bad_score 	= np.sum(deviation_converged[bad_indicator])/bad_indicator.shape[0]
	avg_score 	= np.sum(deviation_converged)/deviation_converged.shape[0]

	print 'Avg_score : %.2f \t Good_score : %.2f \t Bad_score : %.2f' % (avg_score, good_score, bad_score)
	return avg_score, good_score, bad_score

def learning_curve_based_metric(training_log_GAIL, training_log_CVaR, expert_mean, expert_std, fraction, reward_offset):
	# indicator = mean_series>0.8*expert_mean
	# avgDeviation = lines.dot(indicator)/sum(indicator)
	# print "Learning curve based metric: %f"%(avgDeviation/expert_std)
	# # return area

	avg_score_GAIL, good_score_GAIL, bad_score_GAIL = compute_metrics(training_log_GAIL, expert_mean, expert_std, fraction, reward_offset)
	avg_score_CVaR, good_score_CVaR, bad_score_CVaR = compute_metrics(training_log_CVaR, expert_mean, expert_std, fraction, reward_offset)
	print '\nRatios :'
	print 'GAIL \t :: avg : %.2f \t good : %.2f \t bad : %.2f' % (avg_score_GAIL/expert_std, good_score_GAIL/expert_std, bad_score_GAIL/expert_std)
	print 'CVaR \t :: avg : %.2f \t good : %.2f \t bad : %.2f' % (avg_score_CVaR/expert_std, good_score_CVaR/expert_std, bad_score_CVaR/expert_std)
	print '\n'

def mean_metrics(trajectories_GAIL, trajectories_CVaR, mean_exp, reward_offset):
# def mean_metrics(trajectories_expert, trajectories_GAIL, trajectories_CVaR):
	# _,_,r_B_T,_ = load_dataset(trajectories_expert)
	# mean_exp 	= r_B_T.sum(axis=1).mean()
	_,_,r_B_T,_ = load_dataset(trajectories_GAIL)
	mean_GAIL 	= r_B_T.sum(axis=1).mean() + reward_offset
	_,_,r_B_T,_ = load_dataset(trajectories_CVaR)
	mean_CVaR 	= r_B_T.sum(axis=1).mean() + reward_offset
	mean_exp += reward_offset

	print '\nMeans:'
	print 'Expert \t: %.2f' %(mean_exp)
	print 'GAIL \t: %.2f' %(mean_GAIL)
	print 'CVaR \t: %.2f' %(mean_CVaR)
	print '\nRatios:'
	print 'GAIL/exp = %.2f' %(mean_GAIL/mean_exp)
	print 'CVaR/exp = %.2f' %(mean_CVaR/mean_exp)
	print

def plot_single_reward_histogram(trajectories, name):
	_,_,r_B_T,exlen_B = load_dataset(trajectories)
	traj_rewards = r_B_T.sum(axis=1)

	global numTrajs
	numTrajs = len(exlen_B)

	mpl.rcParams['xtick.labelsize'] = 20
	mpl.rcParams['ytick.labelsize'] = 20

	x_min = 0		
	x_max = 10500	# Humanoid
	y_max = 50		# Humanoid

	bin_width = (x_max-x_min)//100
	bins = np.arange(x_min, x_max+bin_width, bin_width)

	plt.figure(1)
	plt.hist(traj_rewards, bins=np.sort(-bins))
	plt.ylim([0,y_max])
	plt.yticks(np.arange(0, y_max+1, (y_max//4)))
	plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
	plt.axvline(x=q_VaRvalue, color='red', linestyle='--')
	plt.plot(x=q_CVaRvalue, y=0, color='g', marker='o')
	# plt.savefig("histogram_of_trajectory_costs_"+name+trajectories.split('/')[-1]+".png")
	plt.show()
	plt.close()

def plot_single_cost_histogram(trajectories, name):
	_,_,r_B_T,exlen_B = load_dataset(trajectories)
	traj_rewards = r_B_T.sum(axis=1)

	mean = traj_rewards.mean()
	sigma = traj_rewards.std()

	global numTrajs
	numTrajs = len(exlen_B)
	print numTrajs

	mpl.rcParams['xtick.labelsize'] = 20
	mpl.rcParams['ytick.labelsize'] = 20
	
	x_min = 0		
	# x_min = -200	# Reacher

	# x_max = 10	# Reacher
	x_max = 10500	# Humanoid
	# x_max = 5500	# Walker
	# x_max = 4000	# Hopper, HalfCheetah
	
	# y_max = 125		# Walker
	y_max = 100		# Humanoid
	# y_max = 200		# Hopper
	# y_max = 100		# Reacher
	# y_max = 75		# HalfCheetah
	
	# model = GMM(n_components=1)
	# model.fit(traj_rewards.reshape(-1,1))
	# mean, sigma = model.means_[0], math.sqrt(model.covariances_[0][0][0])
	x_max_zoomed = int(mean - 2*sigma)
	print "mean : %.2f, sigma : %.2f\nx_max_zoomed : %.2f" % (mean, sigma, x_max_zoomed)

	bin_width = (x_max-x_min)//100
	bins = np.arange(x_min, x_max+bin_width, bin_width)
	bins_zoomed = np.arange(x_min, x_max_zoomed+bin_width, bin_width)
	xticks_zoomed = np.arange(x_min, x_max_zoomed+3, (x_max_zoomed-x_min+3)//4) #if x_max_zoomed>0 else np.arange(x_min, x_max_zoomed-3, abs(x_min-x_max_zoomed)//4)
	print xticks_zoomed

	traj_rewards = -traj_rewards
	x_max, x_min = -x_min, -x_max 		# to convert rewards to costs
	x_min_zoomed = -x_max_zoomed

	q_VaRvalue, q_CVaRvalue	 = evaluate_q_CVaRvalue(trajectories, 0.9)

	plt.figure(1)
	plt.hist(traj_rewards, bins=np.sort(-bins))
	plt.ylim([0,y_max])
	# plt.xlim([-10500, -9500])
	plt.yticks(np.arange(0, y_max+1, (y_max//4)))
	# plt.locator_params(nticks=4)
	# plt.xticks(np.arange(min(traj_rewards), max(traj_rewards)+1, (max(traj_rewards)-min(traj_rewards))//4))
	plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
	plt.axvline(x=q_VaRvalue, color='r', linestyle='--')
	plt.axvline(x=q_CVaRvalue, color='g', linestyle='--')
	# plt.scatter(x=[q_CVaRvalue], y=[0], color='black', marker='x')
	plt.savefig("histogram_of_trajectory_costs_"+name+trajectories.split('/')[-1]+".png")
	# plt.show()
	plt.close()

	## x_max = 3000	# Hopper, HalfCheetah
	## x_max = 4500	# Walker
	## x_max = 8000	# Humanoid
	## x_max = -50	# Reacher
	
	# y_max = 15

	# plt.figure(2)
	# # plt.hist(traj_rewards,bins=100)
	# plt.hist(traj_rewards, bins=np.sort(-bins_zoomed))
	# # plt.xlim([0,x_max])
	# # plt.xlim([x_min-bin_width,0])
	# # xlims = [0,x_min_zoomed] if x_min_zoomed>0 else [x_min_zoomed,0]
	# plt.xlim([x_min_zoomed,x_max]) #if x_min_zoomed<0 else [0,x_min_zoomed])	
	# plt.ylim([0,y_max])
	# # plt.xticks(np.arange(min(traj_rewards), max(traj_rewards)+1, (max(traj_rewards)-min(traj_rewards))//4))
	# plt.xticks(np.sort(-xticks_zoomed))
	# # plt.yticks([i*100/numTrajs for i in range(0, y_max+2, (y_max+1)/4)][1:])
	# plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
	# plt.savefig("histogram_of_trajectory_costs_"+name+trajectories.split('/')[-1]+"_zoomed.png")
	# # plt.show()
	# plt.close()

	# sorted_traj_rewards = np.sort(traj_rewards)
	# print sorted_traj_rewards[-5:], sorted_traj_rewards[-5:].mean()

def plot_all_cost_histograms(trajectories_expert, trajectories_GAIL, trajectories_CVaR):
	# plot_single_cost_histogram(trajectories_expert, '1')
	plot_single_cost_histogram(trajectories_GAIL, '2')
	plot_single_cost_histogram(trajectories_CVaR, '3')

def to_percent(y, position):
	# Ignore the passed in position. This has the effect of scaling the default tick locations.
	s = str(int(y*100/numTrajs))
	# s = str(y)

	# The percent symbol needs escaping in latex
	if mpl.rcParams['text.usetex'] is True:
		return s + r'$\%$'
	else:
		return s + '%'

def overlay_reward_histograms(trajectories_GAIL, trajectories_CVaR, env_name):

	_,_,r_B_T_GAIL,exlen_B = load_dataset(trajectories_GAIL)
	traj_rewards_GAIL = r_B_T_GAIL.sum(axis=1)
	_,_,r_B_T_CVaR,_ = load_dataset(trajectories_CVaR)
	traj_rewards_CVaR = r_B_T_CVaR.sum(axis=1)
	global numTrajs
	numTrajs = len(exlen_B)
	extra = 20

def sample_trajectories(trajectories):

	obs_B_T_Do,a_B_T_Da,r_B_T,len_B = load_dataset(trajectories)
	totalTrajs = np.shape(len_B)[0]

	for numTrajs in [32,64,128,256,512]:

		idx = random.sample(range(totalTrajs), numTrajs)
		file_name = trajectories + '_' + str(numTrajs)

		with h5py.File(file_name, 'w') as f:
			def write(name, a):
				# chunks of 128 trajs each
				f.create_dataset(name, data=a, chunks=(min(128, a.shape[0]),)+a.shape[1:], compression='gzip', compression_opts=9)

			# Right-padded trajectory data
			write('obs_B_T_Do', obs_B_T_Do[idx])
			write('a_B_T_Da', a_B_T_Da[idx])
			write('r_B_T', r_B_T[idx])
			# Trajectory lengths
			write('len_B', len_B[idx])

			# Also save args to this script
			argstr = json.dumps(vars(args), separators=(',', ':'), indent=2)
			f.attrs['args'] = argstr

	# print traj_rewards_CVaR.shape
	traj_rewards_GAIL = -traj_rewards_GAIL
	traj_rewards_CVaR = -traj_rewards_CVaR
	
	bins = np.linspace(min(np.min(traj_rewards_GAIL), np.min(traj_rewards_CVaR)) - extra, max(np.max(traj_rewards_GAIL), np.max(traj_rewards_CVaR)) + extra, 100)

	# print traj_rewards_GAIL

	# # plt.figure()
	plt.hist(traj_rewards_GAIL, bins, alpha=0.5, label='GAIL', color='g')
	plt.hist(traj_rewards_CVaR, bins, alpha=0.5, label='Expert', color='b')
	plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
	# plt.legend(loc='center')
	plt.legend()
	plt.title(env_name, fontsize=16)
	plt.xlabel('Trajectory cost')
	plt.ylabel('Percentage')
	plt.savefig("histogram_of_trajectory_rewards_"+trajectories_GAIL.split('/')[-1][:-3]+"--on--"+trajectories_GAIL.split('/')[-1][:-3]+".png")
	plt.show()

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--trajectories', type=str)#, required=True)
	parser.add_argument('--trajectories_expert', type=str)#, required=True)
	parser.add_argument('--trajectories_GAIL', type=str)#, required=True)
	parser.add_argument('--trajectories_RAIL', type=str)#, required=True)
	parser.add_argument('--policy_GAIL', type=str)#, required=True)
	parser.add_argument('--policy_CVaR', type=str)#, required=True)
	parser.add_argument('--expert_mean', type=float)#, required=True)
	parser.add_argument('--expert_std', type=float)#, required=True)
	parser.add_argument('--fraction', type=float, default=0.8)
	parser.add_argument('--percentile', type=float, default=10)
	parser.add_argument('--num_bins', type=int, default=10)
	parser.add_argument('--env_name', type=str)#, required=True)
	parser.add_argument('--reward_offset', type=int, default=0)
	parser.add_argument('--CVaR_alpha', type=float, default=0.9)
	args = parser.parse_args()

	# # evaluate_q_percentile_traj_reward(args.trajectories)
	# # evaluate_worst_50_percent_population_size(args.trajectories)
	# overlay_reward_histograms(args.trajectories_GAIL, args.trajectories_CVaR, args.env_name)
	# plot_all_cost_histograms(args.trajectories_expert, args.trajectories_GAIL, args.trajectories_CVaR)
	compare_q_CVaR_all3(args.trajectories_expert, args.trajectories_GAIL, args.trajectories_RAIL, args.CVaR_alpha)
	# sample_trajectories(args.trajectories)
	# compare_VaR_CVaR(args.CVaR_alpha)
	# compare_q_percentile_traj_reward(args.trajectories_GAIL, args.trajectories_CVaR, args.percentile)
	# compare_q_CVaR(args.trajectories_GAIL, args.trajectories_CVaR, args.CVaR_alpha)
	# mean_metrics(args.trajectories_GAIL, args.trajectories_CVaR, args.expert_mean, args.reward_offset)
	# learning_curve_based_metric(args.policy_GAIL, args.policy_CVaR, args.expert_mean, args.expert_std, args.fraction, args.reward_offset)
	# compute_variance_wrt_expert(args.policy_GAIL, args.policy_CVaR, args.expert_mean, args.expert_std, args.fraction)
	# compare_leftmost(args.trajectories_GAIL, args.trajectories_CVaR, args.num_bins)
	# load_dataset(args.trajectories)


# python scripts/evaluate.py --trajectories FinalEvaluation/ours/trajectories/ga_501-iter_Walker2d_CVaR_lambda_0.25-trajectories.h5 --policy_CVaR FinalEvaluation/ours/policies/ga_501_iter_Walker2d_CVaR_lambda_0.25_policy.h5 --policy_GAIL FinalEvaluation/vanilla/policies/vanilla_GAIL_Walker2d_policy.h5 --expert_mean 5317.64 --expert_std 655.55

######################################################################################################################

# python scripts/evaluate.py --trajectories_GAIL trajectories/Agent/vanilla_Walker2d_trajectories_460.h5 --trajectories_CVaR trajectories/Agent/ga_501_iter_Walker2d_CVaR_lambda_0.25_trajectories_460.h5 --env_name Walker-v1 --percentile 10 --expert_mean 5317.64 --expert_std 655.55 --env_name Walker --trajectories_expert trajectories/expert_trajectories-Walker2d --CVaR_alpha 0.9

# python scripts/evaluate.py --trajectories_GAIL FinalEvaluation/vanilla/trajectories/vanilla_GAIL_Hopper_trajectories.h5 --trajectories_CVaR FinalEvaluation/ours/trajectories/ga_501-iter_Hopper_CVaR_lambda_0.5-trajectories.h5 --env_name Hopper --percentile 10 --expert_mean 3666.88 --expert_std 392.908 --env_name Hopper --trajectories_expert trajectories/expert_trajectories-Hopper

# python scripts/evaluate.py --trajectories_GAIL FinalEvaluation/vanilla/trajectories/vanilla_GAIL_HalfCheetah_trajectories.h5 --trajectories_CVaR FinalEvaluation/ours/trajectories/ga_501-iter_HalfCheetah-v1_CVaR_lambda_0.5_reproduction-trajectories.h5 --percentile 10 --expert_mean 3583.946 --expert_std 101.54 --env_name HalfCheetah --trajectories_expert trajectories/expert_trajectories-HalfCheetah --CVaR_alpha 0.9

# python scripts/evaluate.py --trajectories_GAIL FinalEvaluation/vanilla/trajectories/humanoid_noCVaR_500.h5 --trajectories_CVaR FinalEvaluation/ours/trajectories/ga_1501-iter_Humanoid-v1_CVaR_lambda_0.75_240_trajs_lr_0.0075_alpha_0.9_trajectories_1500.h5 --env_name Humanoid --CVaR_alpha 0.9 --expert_mean 9790.99 --expert_std 1175.06 --env_name Humanoid --trajectories_expert trajectories/expert_trajectories-Humanoid

# python scripts/evaluate.py --trajectories_GAIL FinalEvaluation/vanilla/trajectories/vanilla_Reacher_trajectories.h5 --trajectories_CVaR FinalEvaluation/ours/trajectories/ga_201-iter_Reacher_CVaR_lambda_0.25_trajectories.h5 --expert_mean -4.08 --expert_std 1.44 --env_name Reacher --CVaR_alpha 0.9 --reward_offset 200 --trajectories_expert trajectories/expert_trajectories-Reacher-v1
######################################################################################################################

# python scripts/evaluate.py --policy_GAIL FinalEvaluation/vanilla/policies/vanilla_MountainCar_2_policy.h5 --policy_CVaR FinalEvaluation/ours/policies/ga_iter_301_MountainCar_CVaR_lambda_0.5_min-sa_5000_policy.h5 --expert_mean -99.8 --expert_std 6.6 --fraction 0.8

# python scripts/evaluate.py --policy_GAIL FinalEvaluation/vanilla/policies/vanilla_Reacher_policy.h5 --policy_CVaR FinalEvaluation/ours/policies/ga_iter_201_Reacher_CVaR_lambda_0.25_policy.h5 --expert_mean -4.08 --expert_std 1.44 --env_name Reacher --fraction 0.8 --reward_offset 200

######################################################################################################################

######################## For the histograms

# python scripts/evaluate.py --trajectories_expert FinalEvaluation/histograms/expert_Hopper_250 --trajectories_GAIL FinalEvaluation/histograms/vanilla_Hopper_250 --trajectories_CVaR FinalEvaluation/histograms/CVaR_Hopper_250

# python scripts/evaluate.py --trajectories_expert FinalEvaluation/histograms/expert_HalfCheetah_250 --trajectories_GAIL FinalEvaluation/histograms/vanilla_HalfCheetah_250 --trajectories_CVaR FinalEvaluation/histograms/CVaR_HalfCheetah_250

# python scripts/evaluate.py --trajectories_expert FinalEvaluation/histograms/expert_Reacher_250 --trajectories_GAIL FinalEvaluation/histograms/vanilla_Reacher_250 --trajectories_CVaR FinalEvaluation/histograms/CVaR_Reacher_250

# python scripts/evaluate.py --trajectories_expert FinalEvaluation/histograms/expert_Walker_250 --trajectories_GAIL FinalEvaluation/histograms/vanilla_Walker_250_iter460 --trajectories_CVaR FinalEvaluation/histograms/CVaR_Walker_250_iter480

# python scripts/evaluate.py --trajectories_expert FinalEvaluation/histograms/expert_Humanoid_250 --trajectories_GAIL FinalEvaluation/histograms/vanilla_Humanoid_250 --trajectories_CVaR FinalEvaluation/histograms/CVaR_Humanoid_250
