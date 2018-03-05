from variance_analysis import * 
import os

def searchForParams(policyFile, expert_trajectories, n_components_gmm, numTrajs, length_filtering_expert_trajs):
	
	print 'Fitting the GMM with %d components on the expert trajectories at %s' % (n_components_gmm, expert_trajectories) 
	gmm = estimate_state_density(expert_trajectories, n_components_gmm, length_filtering_expert_trajs)

	f = h5py.File(policyFile)			# check the location

	# count = 0
	print '\nBeginning the analysis...'
	print '======================================================\n'

	out_file = open('param_search_out_0percent_filtering_ngmm_5.csv','w+')
	
	for policyKey in f['snapshots'].keys()[::10]:

		policyName = policyFile + '/snapshots/' + policyKey
		# print policyName
		outTrajsFile = 'trajectories/HyperparamAnalysis/b20kt50/' + policyKey
		# print 'Generating %d trajectories for policy %s...' % (numTrajs, policyKey)
		# command = 'python scripts/vis_mj.py ' + policyName + ' --count ' + str(numTrajs) + ' --out ' + outTrajsFile 
		# os.system(command)

		print 'Calculating scores...'
		bad_obs_B_T_Do, bad_a_B_T_Da, bad_r_B_T, bad_len_B = find_underperforming_trajectories(outTrajsFile, 10, use_percent=True)
		good_obs_B_T_Do, good_a_B_T_Da, good_r_B_T, good_len_B = find_overperforming_trajectories(outTrajsFile, 90, use_percent=True)
		bad_obs_flattened = bad_obs_B_T_Do.reshape((-1,bad_obs_B_T_Do.shape[-1]))
		good_obs_flattened = good_obs_B_T_Do.reshape((-1,good_obs_B_T_Do.shape[-1]))	
		scores_bad = gmm.score_samples(bad_obs_flattened)
		scores_good = gmm.score_samples(good_obs_flattened)	
		# print "Mean score of good states: %f, bad states: %f" %(scores_good.mean(), scores_bad.mean())
		# print "Std score of good states: %f, bad states: %f" %(scores_good.std(), scores_bad.std())
		# print "Min score of good states: %f, bad states: %f" %(scores_good.min(), scores_bad.min())
		# print "Max score of good states: %f, bad states: %f" %(scores_good.max(), scores_bad.max())	
		# print "Bad percentiles:"
		bad_percentiles = []
		good_percentiles = []
		for i in range(11):
			bad_percentiles.append(np.percentile(scores_bad,i*10.))
		for i in range(11):
			good_percentiles.append(np.percentile(scores_good,i*10.))

		print 'Writing to file...'
		out_file.write(policyKey + ',' + ','.join(str(i) for i in bad_percentiles+good_percentiles) + '\n')

		# print '\n======================================================\n'

		# if count==0:
		# 	break


	# def plotData():

	# 	plt.figure()
	# 	plt.plot(, marker='o',alpha=0.25, facecolors='b')
	# 	# plt.savefig('.png')
	# 	plt.show() 

searchForParams(policyFile='training_logs/additiveStatePrior/early_stopping/b20kt50/conditionalThrashing_b20_kt50_earlyStoppingv2_policy.h5', expert_trajectories='trajectories/expert_trajectories-Humanoid', n_components_gmm=5, numTrajs=50, length_filtering_expert_trajs=False)

print 'Terminated.'

