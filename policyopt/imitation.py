from . import nn, rl, util, RaggedArray, ContinuousSpace, FiniteSpace, optim, thutil, CVaR
import numpy as np
from contextlib import contextmanager
import theano; from theano import tensor

from scipy.optimize import fmin_l_bfgs_b
from sklearn.mixture import GaussianMixture as GMM
import pdb
# import h5py

class BehavioralCloningOptimizer(object):
	def __init__(self, 
		mdp, policy, 
		lr, batch_size, 
		obsfeat_fn, #Feature extraction function
		ex_obs, ex_a, #Example observations and actions
		eval_sim_cfg, eval_freq, train_frac):
		
		self.mdp, self.policy, self.lr, self.batch_size, self.obsfeat_fn = mdp, policy, lr, batch_size, obsfeat_fn

		# Randomly split data into train/val
		assert ex_obs.shape[0] == ex_a.shape[0]
		num_examples = ex_obs.shape[0]
		num_train = int(train_frac * num_examples)
		shuffled_inds = np.random.permutation(num_examples)
		train_inds, val_inds = shuffled_inds[:num_train], shuffled_inds[num_train:]
		assert len(train_inds) >= 1 and len(val_inds) >= 1
		print '{} training examples and {} validation examples'.format(len(train_inds), len(val_inds))
		self.train_ex_obsfeat, self.train_ex_a = self.obsfeat_fn(ex_obs[train_inds]), ex_a[train_inds]
		self.val_ex_obsfeat, self.val_ex_a = self.obsfeat_fn(ex_obs[val_inds]), ex_a[val_inds]

		self.eval_sim_cfg = eval_sim_cfg
		self.eval_freq = eval_freq

		self.total_time = 0.
		self.curr_iter = 0

	def step(self):
		with util.Timer() as t_all:
			# Subsample expert transitions for SGD : rather, choose a random batch of self.batch_size training samples 
			inds = np.random.choice(self.train_ex_obsfeat.shape[0], size=self.batch_size)
			batch_obsfeat_B_Do = self.train_ex_obsfeat[inds,:]
			batch_a_B_Da = self.train_ex_a[inds,:]
			# Take step
			loss = self.policy.step_bclone(batch_obsfeat_B_Do, batch_a_B_Da, self.lr)

			# Roll out trajectories when it's time to evaluate our policy
			val_loss = val_acc = trueret = avgr = ent = np.nan
			avglen = -1
			if self.eval_freq != 0 and self.curr_iter % self.eval_freq == 0:
				val_loss = self.policy.compute_bclone_loss(self.val_ex_obsfeat, self.val_ex_a)
				# Evaluate validation accuracy (independent of standard deviation)
				if isinstance(self.mdp.action_space, ContinuousSpace):
					val_acc = -np.square(self.policy.compute_actiondist_mean(self.val_ex_obsfeat) - self.val_ex_a).sum(axis=1).mean()
				else:
					assert self.val_ex_a.shape[1] == 1
					# val_acc = (self.policy.sample_actions(self.val_ex_obsfeat)[1].argmax(axis=1) == self.val_ex_a[1]).mean()
					val_acc = -val_loss # val accuracy doesn't seem too meaningful so just use this


		# Log
		self.total_time += t_all.dt
		fields = [
			('iter', self.curr_iter, int),
			('bcloss', loss, float), # supervised learning loss
			('valloss', val_loss, float), # loss on validation set
			('valacc', val_acc, float), # loss on validation set
			('trueret', trueret, float), # true average return for this batch of trajectories
			('avgr', avgr, float), # average reward encountered
			('avglen', avglen, int), # average traj length
			('ent', ent, float), # entropy of action distributions
			('ttotal', self.total_time, float), # total time
		]
		self.curr_iter += 1
		return fields


class TransitionClassifier(nn.Model):
	'''Reward/adversary for generative-adversarial training'''

	def __init__(self, 
		obsfeat_space, action_space, 
		hidden_spec, 
		max_kl, 
		adam_lr, adam_steps, 
		ent_reg_weight, 
		enable_inputnorm, 
		include_time, time_scale, # Defined relatively in imitation_mj while instantiating this class as 1./mdp.env_spec.timestep_limit which makes sense
		favor_zero_expert_reward, 
		varscope_name,
		useCVaR=False,
		CVaR_loss_weightage = 1.  
		):
		
		self.obsfeat_space, self.action_space = obsfeat_space, action_space
		self.hidden_spec = hidden_spec #For the discriminator network?
		self.max_kl = max_kl 
		self.adam_steps = adam_steps
		self.ent_reg_weight = ent_reg_weight; assert ent_reg_weight >= 0
		self.include_time = include_time
		self.time_scale = time_scale
		self.favor_zero_expert_reward = favor_zero_expert_reward #What is it?
		self.useCVaR = useCVaR
		self.CVaR_loss_weightage = CVaR_loss_weightage

		with nn.variable_scope(varscope_name) as self.__varscope:
			# Map (s,a) pairs to classifier scores (log probabilities of classes)
			obsfeat_B_Df = tensor.matrix(name='obsfeat_B_Df')
			a_B_Da = tensor.matrix(name='a_B_Da', dtype=theano.config.floatX if self.action_space.storage_type == float else 'int64')
			t_B = tensor.vector(name='t_B')
			
			CVaR_weights = tensor.vector(name='CVaR_weights')

			scaled_t_B = self.time_scale * t_B #Convert the time index into a real number by scaling with time_scale

			if isinstance(self.action_space, ContinuousSpace):
				# For a continuous action space, map observation-action pairs to a real number (reward)
				trans_B_Doa = tensor.concatenate([obsfeat_B_Df, a_B_Da], axis=1)
				trans_dim = self.obsfeat_space.dim + self.action_space.dim
				# Normalize
				with nn.variable_scope('inputnorm'):
					self.inputnorm = (nn.Standardizer if enable_inputnorm else nn.NoOpStandardizer)(self.obsfeat_space.dim + self.action_space.dim)
				normedtrans_B_Doa = self.inputnorm.standardize_expr(trans_B_Doa)
				if self.include_time:
					net_input = tensor.concatenate([normedtrans_B_Doa, scaled_t_B[:,None]], axis=1)
					net_input_dim = trans_dim + 1
				else:
					net_input = normedtrans_B_Doa
					net_input_dim = trans_dim
				# Compute scores
				with nn.variable_scope('hidden'):
					net = nn.FeedforwardNet(net_input, (net_input_dim,), self.hidden_spec)
				with nn.variable_scope('out'):
					out_layer = nn.AffineLayer(net.output, net.output_shape, (1,), initializer=np.zeros((net.output_shape[0], 1),dtype=theano.config.floatX))
				scores_B = out_layer.output[:,0]

			else:
				# For a finite action space, map observation observations to a vector of rewards
				# because feeding the discrete action number to the neural network makes no sense and is not scalable for obvious reasons

				# Normalize observations
				with nn.variable_scope('inputnorm'):
					self.inputnorm = (nn.Standardizer if enable_inputnorm else nn.NoOpStandardizer)(self.obsfeat_space.dim)
				normedobs_B_Df = self.inputnorm.standardize_expr(obsfeat_B_Df)
				if self.include_time:
					net_input = tensor.concatenate([normedobs_B_Df, scaled_t_B[:,None]], axis=1)
					net_input_dim = self.obsfeat_space.dim + 1
				else:
					net_input = normedobs_B_Df
					net_input_dim = self.obsfeat_space.dim
				# Compute scores
				with nn.variable_scope('hidden'):
					net = nn.FeedforwardNet(net_input, (net_input_dim,), self.hidden_spec)
				with nn.variable_scope('out'):
					out_layer = nn.AffineLayer(
						net.output, net.output_shape, (self.action_space.size,),
						initializer=np.zeros((net.output_shape[0], self.action_space.size),dtype=theano.config.floatX))
				scores_B = out_layer.output[tensor.arange(normedobs_B_Df.shape[0]), a_B_Da[:,0]] #Select the scores corresponding to the actions of the batch 


		if self.include_time:
			self._compute_scores = thutil.function([obsfeat_B_Df, a_B_Da, t_B], scores_B) # scores define the conditional distribution p(label | (state,action))
		else:
			compute_scores_without_time = thutil.function([obsfeat_B_Df, a_B_Da], scores_B)
			self._compute_scores = lambda _obsfeat_B_Df, _a_B_Da, _t_B: compute_scores_without_time(_obsfeat_B_Df, _a_B_Da)

		if self.favor_zero_expert_reward:
			# 0 for expert-like states, goes to -inf for non-expert-like states
			# compatible with envs with traj cutoffs for good (expert-like) behavior
			# e.g. mountain car, which gets cut off when the car reaches the destination
			rewards_B = thutil.logsigmoid(scores_B)
		else:
			# 0 for non-expert-like states, goes to +inf for expert-like states
			# compatible with envs with traj cutoffs for bad (non-expert-like) behavior
			# e.g. walking simulations that get cut off when the robot falls over
			rewards_B = -tensor.log(1.-tensor.nnet.sigmoid(scores_B))
		if self.include_time:
			self._compute_reward = thutil.function([obsfeat_B_Df, a_B_Da, t_B], rewards_B)
		else:
			compute_reward_without_time = thutil.function([obsfeat_B_Df, a_B_Da], rewards_B)
			self._compute_reward = lambda _obsfeat_B_Df, _a_B_Da, _t_B: compute_reward_without_time(_obsfeat_B_Df, _a_B_Da)

		param_vars = self.get_trainable_variables()

		# Logistic regression loss, regularized by negative entropy
		labels_B = tensor.vector(name='labels_B') #What are these?
		weights_B = tensor.vector(name='weights_B')
		losses_B = thutil.sigmoid_cross_entropy_with_logits(scores_B, labels_B)
		ent_B = thutil.logit_bernoulli_entropy(scores_B)
		loss_AL = ((losses_B - self.ent_reg_weight*ent_B)*weights_B).sum(axis=0) 
		if self.useCVaR:
			# loss_CVaR = CVaR_weights.mean()*rewards_B.sum() #DANGER: it should have been mean of the sums over traj
			loss_CVaR = CVaR_weights.mean()*rewards_B.mean() #FiXME: upper line better theory compliant
			# loss = loss_AL+loss_CVaR
			loss = loss_AL - self.CVaR_loss_weightage*loss_CVaR #CHECK: Negative sign for maximizing CVaR loss
		else:
			loss = loss_AL

		lossgrad_P = thutil.flatgrad(loss, param_vars) 

		if self.include_time:
			self._adamstep = thutil.function(
				[obsfeat_B_Df, a_B_Da, t_B, labels_B, weights_B, CVaR_weights], [loss, lossgrad_P],
				updates=thutil.adam(loss, param_vars, lr=adam_lr), on_unused_input='ignore') #CVaR_weights may be unused
		else:
			adamstep_without_time = thutil.function(
				[obsfeat_B_Df, a_B_Da, labels_B, weights_B, CVaR_weights], [loss, lossgrad_P],
				updates=thutil.adam(loss, param_vars, lr=adam_lr), on_unused_input='ignore')
			self._adamstep = lambda _obsfeat_B_Df, _a_B_Da, _t_B, _labels_B, _weights_B, _CVaR_weights: adamstep_without_time(_obsfeat_B_Df, _a_B_Da, _labels_B, _weights_B, _CVaR_weights) #CVaR_weights may be unused

	@property
	def varscope(self): return self.__varscope

	def compute_reward(self, obsfeat_B_Df, a_B_Da, t_B):
		return self._compute_reward(obsfeat_B_Df, a_B_Da, t_B)

	def fit(self, obsfeat_B_Df, a_B_Da, t_B, exobs_Bex_Do, exa_Bex_Da, ext_Bex, CVaR_weights=None):
		# Transitions from the current policy go first, then transitions from the expert
		obsfeat_Ball_Df = np.concatenate([obsfeat_B_Df, exobs_Bex_Do])
		a_Ball_Da = np.concatenate([a_B_Da, exa_Bex_Da])
		t_Ball = np.concatenate([t_B, ext_Bex])

		# Update normalization
		self.update_inputnorm(obsfeat_Ball_Df, a_Ball_Da)

		B = obsfeat_B_Df.shape[0] # number of examples from the current policy
		Ball = obsfeat_Ball_Df.shape[0] # Ball - b = num examples from expert

		# Label expert as 1, current policy as 0
		labels_Ball = np.zeros(Ball,dtype=theano.config.floatX)
		labels_Ball[B:] = 1.

		# Evenly weight the loss terms for the expert and the current policy
		weights_Ball = np.zeros(Ball,dtype=theano.config.floatX)
		weights_Ball[:B] = 1./B
		weights_Ball[B:] = 1./(Ball - B); assert len(weights_Ball[B:]) == Ball-B

		# Optimize
		for _ in range(self.adam_steps):
			loss, grad_loss_params = self._adamstep(obsfeat_Ball_Df, a_Ball_Da, t_Ball, labels_Ball, weights_Ball, CVaR_weights)
			kl, num_bt_steps = None, 0
		# print "shape grad_loss_params:", grad_loss_params.shape
		# Evaluate
		scores_Ball = self._compute_scores(obsfeat_Ball_Df, a_Ball_Da, t_Ball); assert scores_Ball.shape == (Ball,)
		accuracy = .5 * (weights_Ball * ((scores_Ball <= 0) == (labels_Ball == 0))).sum()
		accuracy_for_currpolicy = (scores_Ball[:B] <= 0).mean()
		accuracy_for_expert = (scores_Ball[B:] > 0).mean()
		#assert np.allclose(accuracy, .5*(accuracy_for_currpolicy + accuracy_for_expert)) #Not working after cutting down precision to float32 from float64 - hence commented

		return [
			('rloss', loss, float), # reward function fitting loss
			('racc', accuracy, float), # reward function accuracy
			('rgrad', np.absolute(grad_loss_params).max(), float), #Mean gradient 
			# ('raccpi', accuracy_for_currpolicy, float), # reward function accuracy
			# ('raccex', accuracy_for_expert, float), # reward function accuracy
			# ('rkl', kl, float),
			# ('rbt', num_bt_steps, int),
			# ('rpnorm', util.maxnorm(self.get_params()), float),
			# ('snorm', util.maxnorm(scores_Ball), float),
		]

	def update_inputnorm(self, obs_B_Do, a_B_Da):
		if isinstance(self.action_space, ContinuousSpace):
			self.inputnorm.update(np.concatenate([obs_B_Do, a_B_Da], axis=1))
		else:
			self.inputnorm.update(obs_B_Do)

	def plot(self, ax, idx1, idx2, range1, range2, n=100):
		assert len(range1) == len(range2) == 2 and idx1 != idx2
		x, y = np.mgrid[range1[0]:range1[1]:(n+0j), range2[0]:range2[1]:(n+0j)]

		if isinstance(self.action_space, ContinuousSpace):
			points_B_Doa = np.zeros((n*n, self.obsfeat_space.storage_size + self.action_space.storage_size),dtype=theano.config.floatX)
			points_B_Doa[:,idx1] = x.ravel()
			points_B_Doa[:,idx2] = y.ravel()
			obsfeat_B_Df, a_B_Da = points_B_Doa[:,:self.obsfeat_space.storage_size], points_B_Doa[:,self.obsfeat_space.storage_size:]
			assert a_B_Da.shape[1] == self.action_space.storage_size
			t_B = np.zeros(a_B_Da.shape[0],dtype=theano.config.floatX) # XXX make customizable
			z = self.compute_reward(obsfeat_B_Df, a_B_Da, t_B).reshape(x.shape)
		else:
			obsfeat_B_Df = np.zeros((n*n, self.obsfeat_space.storage_size),dtype=theano.config.floatX)
			obsfeat_B_Df[:,idx1] = x.ravel()
			obsfeat_B_Df[:,idx2] = y.ravel()
			a_B_Da = np.zeros((obsfeat_B_Df.shape[0], 1), dtype=np.int32) # XXX make customizable
			t_B = np.zeros(a_B_Da.shape[0],dtype=theano.config.floatX) # XXX make customizable
			z = self.compute_reward(obsfeat_B_Df, a_B_Da, t_B).reshape(x.shape)

		ax.pcolormesh(x, y, z, cmap='viridis')
		ax.contour(x, y, z, levels=np.log(np.linspace(2., 3., 10)))
		# ax.contourf(x, y, z, levels=[np.log(2.), np.log(2.)+.5], alpha=.5) # high-reward region is highlighted


class LinearReward(object):
	# things to keep in mind
	# - continuous vs discrete actions
	# - simplex or l2 ball
	# - input norm
	# - shifting so that 0 == expert or 0 == non-expert

	def __init__(self,
			obsfeat_space, action_space,
			mode, enable_inputnorm, favor_zero_expert_reward,
			include_time,
			time_scale,
			exobs_Bex_Do, exa_Bex_Da, ext_Bex,
			sqscale=.01,
			quadratic_features=False):

		self.obsfeat_space, self.action_space = obsfeat_space, action_space
		assert mode in ['l2ball', 'simplex']
		print 'Linear reward function type: {}'.format(mode)
		self.simplex = mode == 'simplex'
		self.favor_zero_expert_reward = favor_zero_expert_reward
		self.include_time = include_time
		self.time_scale = time_scale
		self.sqscale = sqscale
		self.quadratic_features = quadratic_features
		self.exobs_Bex_Do, self.exa_Bex_Da, self.ext_Bex = exobs_Bex_Do, exa_Bex_Da, ext_Bex
		with nn.variable_scope('inputnorm'):
			# Standardize both observations and actions if actions are continuous
			# otherwise standardize observations only.
			self.inputnorm = (nn.Standardizer if enable_inputnorm else nn.NoOpStandardizer)(
				(obsfeat_space.dim + action_space.dim) if isinstance(action_space, ContinuousSpace)
					else obsfeat_space.dim)
			self.inputnorm_updated = False
		self.update_inputnorm(self.exobs_Bex_Do, self.exa_Bex_Da) # pre-standardize with expert data

		# Expert feature expectations
		self.expert_feat_Df = self._compute_featexp(self.exobs_Bex_Do, self.exa_Bex_Da, self.ext_Bex)
		# The current reward function
		feat_dim = self.expert_feat_Df.shape[0]
		print 'Linear reward: {} features'.format(feat_dim)
		if self.simplex:
			# widx is the index of the most discriminative reward function
			self.widx = np.random.randint(feat_dim)
		else:
			# w is a weight vector
			self.w = np.random.randn(feat_dim,dtype=theano.config.floatX)
			self.w /= np.linalg.norm(self.w) + 1e-8

		self.reward_bound = 0.

	def _featurize(self, obsfeat_B_Do, a_B_Da, t_B):
		assert self.inputnorm_updated
		assert obsfeat_B_Do.shape[0] == a_B_Da.shape[0] == t_B.shape[0]
		B = obsfeat_B_Do.shape[0]

		# Standardize observations and actions
		if isinstance(self.action_space, ContinuousSpace):
			trans_B_Doa = self.inputnorm.standardize(np.concatenate([obsfeat_B_Do, a_B_Da], axis=1))
			obsfeat_B_Do, a_B_Da = trans_B_Doa[:,:obsfeat_B_Do.shape[1]], trans_B_Doa[:,obsfeat_B_Do.shape[1]:]
			assert obsfeat_B_Do.shape[1] == self.obsfeat_space.dim and a_B_Da.shape[1] == self.action_space.dim
		else:
			assert a_B_Da.shape[1] == 1 and np.allclose(a_B_Da, a_B_Da.astype(int)), 'actions must all be ints'
			obsfeat_B_Do = self.inputnorm.standardize(obsfeat_B_Do)

		# Concatenate with other stuff to get final features
		scaledt_B_1 = t_B[:,None]*self.time_scale
		if isinstance(self.action_space, ContinuousSpace):
			if self.quadratic_features:
				feat_cols = [obsfeat_B_Do, a_B_Da]
				if self.include_time:
					feat_cols.extend([scaledt_B_1])
				feat = np.concatenate(feat_cols, axis=1)
				quadfeat = (feat[:,:,None] * feat[:,None,:]).reshape((B,-1))
				feat_B_Df = np.concatenate([feat,quadfeat,np.ones((B,1),dtype=theano.config.floatX)], axis=1)
			else:
				feat_cols = [obsfeat_B_Do, a_B_Da, (self.sqscale*obsfeat_B_Do)**2, (self.sqscale*a_B_Da)**2]
				if self.include_time:
					feat_cols.extend([scaledt_B_1, scaledt_B_1**2, scaledt_B_1**3])
				feat_cols.append(np.ones((B,1)))
				feat_B_Df = np.concatenate(feat_cols, axis=1)

		else:
			assert not self.quadratic_features
			# Observation-only features
			obsonly_feat_cols = [obsfeat_B_Do, (.01*obsfeat_B_Do)**2]
			if self.include_time:
				obsonly_feat_cols.extend([scaledt_B_1, scaledt_B_1**2, scaledt_B_1**3])
			obsonly_feat_B_f = np.concatenate(obsonly_feat_cols, axis=1)

			# To get features that include actions, we'll have blocks of obs-only features,
			# one block for each action.
			assert a_B_Da.shape[1] == 1
			action_inds = [np.flatnonzero(a_B_Da[:,0] == a) for a in xrange(self.action_space.size)]
			assert sum(len(inds) for inds in action_inds) == B
			action_block_size = obsonly_feat_B_f.shape[1]
			# Place obs features into their appropriate blocks
			blocked_feat_B_Dfm1 = np.zeros((obsonly_feat_B_f.shape[0], action_block_size*self.action_space.size),dtype=theano.config.floatX)
			for a in range(self.action_space.size):
				blocked_feat_B_Dfm1[action_inds[a],a*action_block_size:(a+1)*action_block_size] = obsonly_feat_B_f[action_inds[a],:]
			assert np.isfinite(blocked_feat_B_Dfm1).all()
			feat_B_Df = np.concatenate([blocked_feat_B_Dfm1, np.ones((B,1),dtype=theano.config.floatX)], axis=1)

		if self.simplex:
			feat_B_Df = np.concatenate([feat_B_Df, -feat_B_Df], axis=1)

		assert feat_B_Df.ndim == 2 and feat_B_Df.shape[0] == B
		return feat_B_Df


	def _compute_featexp(self, obsfeat_B_Do, a_B_Da, t_B):
		return self._featurize(obsfeat_B_Do, a_B_Da, t_B).mean(axis=0)


	def fit(self, obsfeat_B_Do, a_B_Da, t_B, _unused_exobs_Bex_Do, _unused_exa_Bex_Da, _unused_ext_Bex):
		# Ignore expert data inputs here, we'll use the one provided in the constructor.

		# Current feature expectations
		curr_feat_Df = self._compute_featexp(obsfeat_B_Do, a_B_Da, t_B)

		# Compute adversary reward
		if self.simplex:
			v = curr_feat_Df - self.expert_feat_Df
			self.widx = np.argmin(v)
			return [('vmin', v.min(), float)]
		else:
			self.w = self.expert_feat_Df - curr_feat_Df
			l2 = np.linalg.norm(self.w)
			self.w /= l2 + 1e-8
			return [('l2', l2, float)]


	def compute_reward(self, obsfeat_B_Do, a_B_Da, t_B):
		feat_B_Df = self._featurize(obsfeat_B_Do, a_B_Da, t_B)
		r_B = (feat_B_Df[:,self.widx] if self.simplex else feat_B_Df.dot(self.w)) / float(feat_B_Df.shape[1])
		assert r_B.shape == (obsfeat_B_Do.shape[0],)

		if self.favor_zero_expert_reward:
			self.reward_bound = max(self.reward_bound, r_B.max())
		else:
			self.reward_bound = min(self.reward_bound, r_B.min())
		shifted_r_B = r_B - self.reward_bound
		if self.favor_zero_expert_reward:
			assert (shifted_r_B <= 0).all()
		else:
			assert (shifted_r_B >= 0).all()

		return shifted_r_B

	def update_inputnorm(self, obs_B_Do, a_B_Da):
		if isinstance(self.action_space, ContinuousSpace):
			self.inputnorm.update(np.concatenate([obs_B_Do, a_B_Da], axis=1))
		else:
			self.inputnorm.update(obs_B_Do)
		self.inputnorm_updated = True


class ImitationOptimizer(object):
	def __init__(self, mdp, discount, lam, policy, sim_cfg, step_func, reward_func, value_func, policy_obsfeat_fn, reward_obsfeat_fn, policy_ent_reg, ex_obs, ex_a, ex_t):
		self.mdp, self.discount, self.lam, self.policy = mdp, discount, lam, policy
		self.sim_cfg = sim_cfg
		self.step_func = step_func
		self.reward_func = reward_func
		self.value_func = value_func
		# assert value_func is not None, 'not tested'
		self.policy_obsfeat_fn = policy_obsfeat_fn
		self.reward_obsfeat_fn = reward_obsfeat_fn
		self.policy_ent_reg = policy_ent_reg
		util.header('Policy entropy regularization: {}'.format(self.policy_ent_reg))

		assert ex_obs.ndim == ex_a.ndim == 2 and ex_t.ndim == 1 and ex_obs.shape[0] == ex_a.shape[0] == ex_t.shape[0]
		self.ex_pobsfeat, self.ex_robsfeat, self.ex_a, self.ex_t = policy_obsfeat_fn(ex_obs), reward_obsfeat_fn(ex_obs), ex_a, ex_t

		self.total_num_trajs = 0
		self.total_num_sa = 0
		self.total_time = 0.
		self.curr_iter = 0
		self.last_sampbatch = None # for outside access for debugging

	def step(self, iter): # All training is done by this function

		with util.Timer() as t_all:

			# Sample trajectories using current policy
			# print 'Sampling'
			with util.Timer() as t_sample:
				try:
					sampbatch = self.mdp.sim_mp(
						policy_fn=lambda obsfeat_B_Df: self.policy.sample_actions(obsfeat_B_Df),
						obsfeat_fn=self.policy_obsfeat_fn,
						cfg=self.sim_cfg)
				except AttributeError:
					pass
				samp_pobsfeat = sampbatch.obsfeat
				self.last_sampbatch = sampbatch
				# print "Sampled batchsize:\n", sampbatch.__len__(), " ---End"

			# Compute baseline / advantages
			# print 'Computing advantages'
			with util.Timer() as t_adv:
				# Compute observation features for reward input
				samp_robsfeat_stacked = self.reward_obsfeat_fn(sampbatch.obs.stacked)
				# Reward is computed wrt current reward function
				# TODO: normalize rewards
				rcurr_stacked = self.reward_func.compute_reward(samp_robsfeat_stacked, sampbatch.a.stacked, sampbatch.time.stacked)
				assert rcurr_stacked.shape == (samp_robsfeat_stacked.shape[0],)

				# If we're regularizing the policy, add negative log probabilities to the rewards
				# Intuitively, the policy gets a bonus for being less certain of its actions
				orig_rcurr_stacked = rcurr_stacked.copy()
				if self.policy_ent_reg is not None and self.policy_ent_reg != 0:
					assert self.policy_ent_reg > 0
					# XXX probably faster to compute this from sampbatch.adist instead
					actionlogprobs_B = self.policy.compute_action_logprobs(samp_pobsfeat.stacked, sampbatch.a.stacked)
					policyentbonus_B = -self.policy_ent_reg * actionlogprobs_B
					rcurr_stacked += policyentbonus_B
				else:
					policyentbonus_B = np.zeros_like(rcurr_stacked,dtype=theano.config.floatX)

				rcurr = RaggedArray(rcurr_stacked, lengths=sampbatch.r.lengths)
				# print "Returned reward shape: ", len(rcurr.arrays) 

				# Compute advantages using these rewards 
				# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> qval[traj_id][0] gives traj cost >>>>>>>>>>>>>> 
				advantages, qvals, vfunc_r2, simplev_r2 = rl.compute_advantage(
					rcurr, samp_pobsfeat, sampbatch.time, self.value_func, self.discount, self.lam)
				



			# >>>>>>>>>>>>>>>>>>> Take a step <<<<<<<<<<<<<<<<<<<<

			# print 'Fitting policy'
			with util.Timer() as t_step:

				# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Policy parameters are extracted here <<<<<<<<<<<<<<<<<<<<<<<<<<<
				params0_P = self.policy.get_params()
				# print "Num Policy params: ", len(params0_P)
				# objgrad_P_test = self.policy.gradients_for_debugging(samp_pobsfeat.stacked, sampbatch.a.stacked, sampbatch.adist.stacked,
				#     advantages.stacked)
				# print "Len of gradient of obj wrt policy params: ", len(objgrad_P_test)

				step_print = self.step_func(
					self.policy, params0_P,
					samp_pobsfeat.stacked, sampbatch.a.stacked, sampbatch.adist.stacked,
					advantages.stacked)

				self.policy.update_obsnorm(samp_pobsfeat.stacked)

			# Fit reward function
			# print 'Fitting reward'
			with util.Timer() as t_r_fit:
				if True:#self.curr_iter % 20 == 0:
					# Subsample expert transitions to the same sample count for the policy
					inds = np.random.choice(self.ex_robsfeat.shape[0], size=samp_pobsfeat.stacked.shape[0])
					exbatch_robsfeat = self.ex_robsfeat[inds,:]
					exbatch_pobsfeat = self.ex_pobsfeat[inds,:] # only used for logging
					exbatch_a = self.ex_a[inds,:]
					exbatch_t = self.ex_t[inds]
					rfit_print = self.reward_func.fit(samp_robsfeat_stacked, sampbatch.a.stacked, sampbatch.time.stacked, exbatch_robsfeat, exbatch_a, exbatch_t)
				else:
					rfit_print = []

			# Fit value function for next iteration
			# print 'Fitting value function'
			with util.Timer() as t_vf_fit:
				if self.value_func is not None:
					# Recompute q vals # XXX: this is only necessary if fitting reward after policy
					# qnew = qvals

					# TODO: this should be a byproduct of reward fitting
					rnew = RaggedArray(
						self.reward_func.compute_reward(samp_robsfeat_stacked, sampbatch.a.stacked, sampbatch.time.stacked),
						lengths=sampbatch.r.lengths)
					qnew, _ = rl.compute_qvals(rnew, self.discount)
					vfit_print = self.value_func.fit(samp_pobsfeat.stacked, sampbatch.time.stacked, qnew.stacked)
				else:
					vfit_print = []



		# Log
		self.total_num_trajs += len(sampbatch)
		self.total_num_sa += sum(len(traj) for traj in sampbatch)
		self.total_time += t_all.dt
		fields = [
			('iter', self.curr_iter, int),
			('trueret', sampbatch.r.padded(fill=0.).sum(axis=1).mean(), float), # average return for this batch of trajectories
			('iret', rcurr.padded(fill=0.).sum(axis=1).mean(), float),
			('trueret_std', sampbatch.r.padded(fill=0.).sum(axis=1).std(), float),
			('ire_std', rcurr.padded(fill=0.).sum(axis=1).std(), float)]
		# fields = [
		#     ('iter', self.curr_iter, int),
		#     ('trueret', sampbatch.r.padded(fill=0.).sum(axis=1).mean(), float), # average return for this batch of trajectories
		#     ('iret', rcurr.padded(fill=0.).sum(axis=1).mean(), float), # average return on imitation reward
		#     ('avglen', int(np.mean([len(traj) for traj in sampbatch])), int), # average traj length
		#     ('ntrajs', self.total_num_trajs, int), # total number of trajs sampled over the course of training
		#     ('nsa', self.total_num_sa, int), # total number of state-action pairs sampled over the course of training
		#     ('ent', self.policy._compute_actiondist_entropy(sampbatch.adist.stacked).mean(), float), # entropy of action distributions
		#     ('vf_r2', vfunc_r2, float),
		#     ('tdvf_r2', simplev_r2, float),
		#     ('dx', util.maxnorm(params0_P - self.policy.get_params()), float), # max parameter difference from last iteration
		# ] + step_print + vfit_print + rfit_print + [
		#     ('avgr', rcurr_stacked.mean(), float), # average regularized reward encountered
		#     ('avgunregr', orig_rcurr_stacked.mean(), float), # average unregularized reward
		#     ('avgpreg', policyentbonus_B.mean(), float), # average policy regularization
		#     # ('bcloss', -self.policy.compute_action_logprobs(exbatch_pobsfeat, exbatch_a).mean(), float), # negative log likelihood of expert actions
		#     # ('bcloss', np.square(self.policy.compute_actiondist_mean(exbatch_pobsfeat) - exbatch_a).sum(axis=1).mean(axis=0), float),
		#     ('tsamp', t_sample.dt, float), # time for sampling
		#     ('tadv', t_adv.dt + t_vf_fit.dt, float), # time for advantage computation
		#     ('tstep', t_step.dt, float), # time for step computation
		#     ('ttotal', self.total_time, float), # total time
		# ]
		fields.extend(step_print)
		fields.extend(rfit_print)
		self.curr_iter += 1
		return fields


class ImitationOptimizer_additiveStatePrior(object):
	def __init__(self, mdp, discount, lam, policy, sim_cfg, step_func, reward_func, value_func, policy_obsfeat_fn, reward_obsfeat_fn, policy_ent_reg, ex_obs, ex_a, ex_t, n_gmm_components, cov_type_gmm, additiveStatePrior_weight, alpha, beta, kickThreshold_percentile, offset=0):#, analysisFile):
		self.mdp, self.discount, self.lam, self.policy = mdp, discount, lam, policy
		self.sim_cfg = sim_cfg
		self.step_func = step_func
		self.reward_func = reward_func
		self.value_func = value_func
		# assert value_func is not None, 'not tested'
		self.policy_obsfeat_fn = policy_obsfeat_fn
		self.reward_obsfeat_fn = reward_obsfeat_fn
		self.policy_ent_reg = policy_ent_reg
		util.header('Policy entropy regularization: {}'.format(self.policy_ent_reg))

		assert ex_obs.ndim == ex_a.ndim == 2 and ex_t.ndim == 1 and ex_obs.shape[0] == ex_a.shape[0] == ex_t.shape[0]
		self.ex_pobsfeat, self.ex_robsfeat, self.ex_a, self.ex_t = policy_obsfeat_fn(ex_obs), reward_obsfeat_fn(ex_obs), ex_a, ex_t

		self.total_num_trajs = 0
		self.total_num_sa = 0
		self.total_time = 0.
		self.curr_iter = offset
		self.last_sampbatch = None # for outside access for debugging
		# for additiveStatePrior
		self.statePrior_model = self.estimate_state_density(ex_obs, n_gmm_components, cov_type_gmm)
		self.additiveStatePrior_weight = additiveStatePrior_weight
		self.alpha = alpha
		self.beta = beta 
		self.kickThreshold_percentile = kickThreshold_percentile
		# self.analysisFile = 'training_logs/additiveStatePrior/Analysis/' + analysisFile

	def step(self, stepNum, kickStatesData): # All training is done by this function

		with util.Timer() as t_all:

			# Sample trajectories using current policy
			# print 'Sampling'
			with util.Timer() as t_sample:
				try:
					sampbatch = self.mdp.sim_mp(
						policy_fn=lambda obsfeat_B_Df: self.policy.sample_actions(obsfeat_B_Df),
						obsfeat_fn=self.policy_obsfeat_fn,
						cfg=self.sim_cfg)
				except AttributeError:
					pass
				samp_pobsfeat = sampbatch.obsfeat
				self.last_sampbatch = sampbatch
				# print "Sampled batchsize:\n", sampbatch.__len__(), " ---End"

			# Compute baseline / advantages
			# print 'Computing advantages'
			with util.Timer() as t_adv:
				# Compute observation features for reward input
				samp_robsfeat_stacked = self.reward_obsfeat_fn(sampbatch.obs.stacked)
				# Reward is computed wrt current reward function
				# TODO: normalize rewards
				rcurr_stacked = self.reward_func.compute_reward(samp_robsfeat_stacked, sampbatch.a.stacked, sampbatch.time.stacked)
				assert rcurr_stacked.shape == (samp_robsfeat_stacked.shape[0],)

				# pdb.set_trace()



				# >>>>>>>>>>>>>>>>>>>Calculate statePrior<<<<<<<<<<<<<<<<<<<<<
				statePrior_stacked = self.familiarity_shaping(self.statePrior_model.score_samples(sampbatch.obs.stacked), self.alpha, self.beta)
				assert statePrior_stacked.shape == (samp_robsfeat_stacked.shape[0],)
				statePrior_stacked_huge_negative = statePrior_stacked<(-self.beta/2.)

				# print "Max reward: %f, min reward: %f" %(rcurr_stacked.max(), rcurr_stacked.min())

				# Synthesize the new reward
				# rcurr_stacked = (1-self.additiveStatePrior_weight)*rcurr_stacked + self.additiveStatePrior_weight*statePrior_stacked
				startGrowth = 800
				endGrowth = 1200

				if stepNum < startGrowth or stepNum > endGrowth:
					self.additiveStatePrior_weight = 0.0
				# elif stepNum < endGrowth:
				# 	self.additiveStatePrior_weight = (stepNum - startGrowth)/(1.0 * (endGrowth-startGrowth))
				else:
					self.additiveStatePrior_weight = 1.0

				
				if self.additiveStatePrior_weight > 0.0:

					# kickThreshold_percentile = 50.0
					kickThreshold_length = np.percentile(sampbatch.time.lengths, self.kickThreshold_percentile)

					traj_lengths_tiled = np.repeat(sampbatch.time.lengths, sampbatch.time.lengths)
					penalized_state_ids = traj_lengths_tiled<kickThreshold_length


					############ For analysis of the kicked-states #############

					# if stepNum==startGrowth:
					# 	with h5py.File(self.analysisFile, 'w') as h5File:
					# 		dataset = h5File.create_dataset('kickStates', (len(data),3), maxshape=(None,3), chunks=True)#, maxshape=(None,), dtype='i8', chunks=(10**4,))
					# 		dataset = np.asarray(data)
					# else:
					# 	with h5py.File(self.analysisFile, 'a') as h5File:
					# 		dataset = h5File['kickStates'] 
					# 		# pdb.set_trace()
					# 		dataset.resize(dataset.shape[0]+len(data), axis=0)
					# 		dataset[-len(data):] = np.asarray(data)
					############################################################

					# flag=0
					# if stepNum > kickThreshold_length:
					# 	flag=1
					 
					rcurr_stacked = rcurr_stacked + self.additiveStatePrior_weight*np.multiply(statePrior_stacked, np.asarray(penalized_state_ids))
					
					indices = np.multiply(statePrior_stacked_huge_negative, penalized_state_ids)
					data = zip(sampbatch.obs.stacked[indices], sampbatch.time.stacked[indices], traj_lengths_tiled[indices])
					# print '\n\n*************************************'
					# print len(data), len(kickStatesData)
					# print '*************************************\n\n'
					kickStatesData.append(data) 
				
				# Causal-entropy regularization
				# If we're regularizing the policy, add negative log probabilities to the rewards
				# Intuitively, the policy gets a bonus for being less certain of its actions
				orig_rcurr_stacked = rcurr_stacked.copy()
				if self.policy_ent_reg is not None and self.policy_ent_reg != 0:
					assert self.policy_ent_reg > 0
					# XXX probably faster to compute this from sampbatch.adist instead
					actionlogprobs_B = self.policy.compute_action_logprobs(samp_pobsfeat.stacked, sampbatch.a.stacked)
					policyentbonus_B = -self.policy_ent_reg * actionlogprobs_B
					rcurr_stacked += policyentbonus_B
				else:
					policyentbonus_B = np.zeros_like(rcurr_stacked,dtype=theano.config.floatX)

				rcurr = RaggedArray(rcurr_stacked, lengths=sampbatch.r.lengths)
				# print "Returned reward shape: ", len(rcurr.arrays) 

				# Compute advantages using these rewards 
				# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> qval[traj_id][0] gives traj cost >>>>>>>>>>>>>> 
				advantages, qvals, vfunc_r2, simplev_r2 = rl.compute_advantage(
					rcurr, samp_pobsfeat, sampbatch.time, self.value_func, self.discount, self.lam)
				



			# >>>>>>>>>>>>>>>>>>> Take a step <<<<<<<<<<<<<<<<<<<<

			# print 'Fitting policy'
			with util.Timer() as t_step:

				# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Policy parameters are extracted here <<<<<<<<<<<<<<<<<<<<<<<<<<<
				params0_P = self.policy.get_params()
				# print "Num Policy params: ", len(params0_P)
				# objgrad_P_test = self.policy.gradients_for_debugging(samp_pobsfeat.stacked, sampbatch.a.stacked, sampbatch.adist.stacked,
				#     advantages.stacked)
				# print "Len of gradient of obj wrt policy params: ", len(objgrad_P_test)

				step_print = self.step_func(
					self.policy, params0_P,
					samp_pobsfeat.stacked, sampbatch.a.stacked, sampbatch.adist.stacked,
					advantages.stacked)

				self.policy.update_obsnorm(samp_pobsfeat.stacked)

			# Fit reward function
			# print 'Fitting reward'
			with util.Timer() as t_r_fit:
				if True:#self.curr_iter % 20 == 0:
					# Subsample expert transitions to the same sample count for the policy
					inds = np.random.choice(self.ex_robsfeat.shape[0], size=samp_pobsfeat.stacked.shape[0])
					exbatch_robsfeat = self.ex_robsfeat[inds,:]
					exbatch_pobsfeat = self.ex_pobsfeat[inds,:] # only used for logging
					exbatch_a = self.ex_a[inds,:]
					exbatch_t = self.ex_t[inds]
					rfit_print = self.reward_func.fit(samp_robsfeat_stacked, sampbatch.a.stacked, sampbatch.time.stacked, exbatch_robsfeat, exbatch_a, exbatch_t)
				else:
					rfit_print = []

			# Fit value function for next iteration
			# print 'Fitting value function'
			with util.Timer() as t_vf_fit:
				if self.value_func is not None:
					# Recompute q vals # XXX: this is only necessary if fitting reward after policy
					# qnew = qvals

					# TODO: this should be a byproduct of reward fitting
					rnew = RaggedArray(
						self.reward_func.compute_reward(samp_robsfeat_stacked, sampbatch.a.stacked, sampbatch.time.stacked),
						lengths=sampbatch.r.lengths)
					qnew, _ = rl.compute_qvals(rnew, self.discount)
					vfit_print = self.value_func.fit(samp_pobsfeat.stacked, sampbatch.time.stacked, qnew.stacked)
				else:
					vfit_print = []



		# Log
		self.total_num_trajs += len(sampbatch)
		self.total_num_sa += sum(len(traj) for traj in sampbatch)
		self.total_time += t_all.dt
		fields = [
			('iter', self.curr_iter, int),
			('trueret', sampbatch.r.padded(fill=0.).sum(axis=1).mean(), float), # average return for this batch of trajectories
			('iret', rcurr.padded(fill=0.).sum(axis=1).mean(), float),
			('trueret_std', sampbatch.r.padded(fill=0.).sum(axis=1).std(), float),
			('ire_std', rcurr.padded(fill=0.).sum(axis=1).std(), float),
			('mean_r+', statePrior_stacked.mean(), float),
			('neg_r_freq', statePrior_stacked_huge_negative.sum(), int),
			('tot_s', statePrior_stacked_huge_negative.shape[0], int),
			('r_mean', orig_rcurr_stacked.mean(), float),
			('r_std', orig_rcurr_stacked.std(), float),
			('r_min', orig_rcurr_stacked.min(), float),
			('r_max', orig_rcurr_stacked.max(), float),
			# ('flag', flag, int),
		# fields = [
		#     ('iter', self.curr_iter, int),
		#     ('trueret', sampbatch.r.padded(fill=0.).sum(axis=1).mean(), float), # average return for this batch of trajectories
		#     ('iret', rcurr.padded(fill=0.).sum(axis=1).mean(), float), # average return on imitation reward
		#     ('avglen', int(np.mean([len(traj) for traj in sampbatch])), int), # average traj length
		#     ('ntrajs', self.total_num_trajs, int), # total number of trajs sampled over the course of training
		#     ('nsa', self.total_num_sa, int), # total number of state-action pairs sampled over the course of training
		#     ('ent', self.policy._compute_actiondist_entropy(sampbatch.adist.stacked).mean(), float), # entropy of action distributions
		#     ('vf_r2', vfunc_r2, float),
		#     ('tdvf_r2', simplev_r2, float),
		#     ('dx', util.maxnorm(params0_P - self.policy.get_params()), float), # max parameter difference from last iteration
		# ] + step_print + vfit_print + rfit_print + [
		#     ('avgr', rcurr_stacked.mean(), float), # average regularized reward encountered
		#     ('avgunregr', orig_rcurr_stacked.mean(), float), # average unregularized reward
		#     ('avgpreg', policyentbonus_B.mean(), float), # average policy regularization
		#     # ('bcloss', -self.policy.compute_action_logprobs(exbatch_pobsfeat, exbatch_a).mean(), float), # negative log likelihood of expert actions
		#     # ('bcloss', np.square(self.policy.compute_actiondist_mean(exbatch_pobsfeat) - exbatch_a).sum(axis=1).mean(axis=0), float),
		#     ('tsamp', t_sample.dt, float), # time for sampling
		#     ('tadv', t_adv.dt + t_vf_fit.dt, float), # time for advantage computation
		#     ('tstep', t_step.dt, float), # time for step computation
		#     ('ttotal', self.total_time, float), # total time
			 # ('tsamp', t_sample.dt, float), # time for sampling
			 # ('tadv', t_adv.dt + t_vf_fit.dt, float), # time for advantage computation
			 # ('t_pstep', t_step.dt, float), # time for step computation
			 # ('t_rfit', t_r_fit.dt, float),
			 # ('t_vffit', t_vf_fit.dt, float),
			 # ('ttotal', self.total_time, float), # total time
		]
		
		fields.extend(step_print)
		fields.extend(rfit_print)
		self.curr_iter += 1
		return fields

	def estimate_state_density(self, ex_obs, n_components, covariance_type):
		model = GMM(n_components=n_components, covariance_type=covariance_type)
		model.fit(ex_obs)
		return model 

	def familiarity_shaping(self, gmm_scores, alpha, beta):
		return -beta*(1/(1+np.exp(gmm_scores+alpha)))



class ImitationOptimizer_CVaR(object):
	def __init__(self, mdp, discount, lam, policy, sim_cfg, step_func, reward_func, value_func, policy_obsfeat_fn, reward_obsfeat_fn, policy_ent_reg, ex_obs, ex_a, ex_t, CVaR_alpha, CVaR_beta, CVaR_lr, CVaR_Lambda_trainable, CVaR_Lambda_val_if_not_trainable, offset=0):
		self.mdp, self.discount, self.lam, self.policy = mdp, discount, lam, policy
		self.sim_cfg = sim_cfg
		self.step_func = step_func
		self.reward_func = reward_func
		self.value_func = value_func
		# assert value_func is not None, 'not tested'
		self.policy_obsfeat_fn = policy_obsfeat_fn
		self.reward_obsfeat_fn = reward_obsfeat_fn
		self.policy_ent_reg = policy_ent_reg
		# CVaR hyperparameters
		self.CVaR_alpha = CVaR_alpha
		self.CVaR_beta = CVaR_beta
		self.CVaR_lr = CVaR_lr
		self.CVaR_Lambda_trainable = CVaR_Lambda_trainable
		self.CVaR_Lambda_val_if_not_trainable = CVaR_Lambda_val_if_not_trainable


		util.header('Policy entropy regularization: {}'.format(self.policy_ent_reg))

		assert ex_obs.ndim == ex_a.ndim == 2 and ex_t.ndim == 1 and ex_obs.shape[0] == ex_a.shape[0] == ex_t.shape[0]
		self.ex_pobsfeat, self.ex_robsfeat, self.ex_a, self.ex_t = policy_obsfeat_fn(ex_obs), reward_obsfeat_fn(ex_obs), ex_a, ex_t

		self.total_num_trajs = 0
		self.total_num_sa = 0
		self.total_time = 0.
		self.curr_iter = offset
		self.last_sampbatch = None # for outside access for debugging

		self.CVaR = CVaR.CVaR_params(self.CVaR_Lambda_trainable, self.CVaR_Lambda_val_if_not_trainable)

	def step(self, iter): # All training is done by this function

		with util.Timer() as t_all:

			# Sample trajectories using current policy
			# print 'Sampling'
			with util.Timer() as t_sample:
				sampbatch = self.mdp.sim_mp(
					policy_fn=lambda obsfeat_B_Df: self.policy.sample_actions(obsfeat_B_Df),
					obsfeat_fn=self.policy_obsfeat_fn,
					cfg=self.sim_cfg)
				samp_pobsfeat = sampbatch.obsfeat
				self.last_sampbatch = sampbatch
				# print "Sampled batchsize:\n", sampbatch.__len__(), " ---End"

			# ============================ Calculate some functions for gradients related to CVaR  ================================
			traj_costs_placeholder = tensor.vector(name='traj_costs')
			D_ge_nu = traj_costs_placeholder>=self.CVaR.nu 
			difference_D_nu = traj_costs_placeholder-self.CVaR.nu 
			policy_param_weights_trajspecific = (self.CVaR.Lambda/(1-self.CVaR_alpha))*D_ge_nu*difference_D_nu
			geometric_sum_discount_placeholder = tensor.vector(name='geometric_sum_discount') 
			reward_param_weights_trajspecific = (self.CVaR.Lambda/(1-self.CVaR_alpha))*D_ge_nu*geometric_sum_discount_placeholder
			nu_gradient = self.CVaR.Lambda*(1 - (1/(1-self.CVaR_alpha))*D_ge_nu.mean())
			Lambda_gradient = self.CVaR.nu - self.CVaR_beta + (1/(1-self.CVaR_alpha))*(D_ge_nu*difference_D_nu).mean()

			get_policyparam_CVaR_weights = theano.function(inputs=[traj_costs_placeholder], outputs=policy_param_weights_trajspecific)
			get_rewardparam_CVaR_weights = theano.function(inputs=[traj_costs_placeholder, geometric_sum_discount_placeholder], outputs=reward_param_weights_trajspecific)
			get_CVaR_nu_gradient = theano.function(inputs=[traj_costs_placeholder], outputs=nu_gradient)
			get_CVaR_Lambda_gradient = theano.function(inputs=[traj_costs_placeholder], outputs=Lambda_gradient)
			# =====================================================================================================================



			# Compute baseline / advantages
			# print 'Computing advantages'
			with util.Timer() as t_adv:
				# Compute observation features for reward input
				samp_robsfeat_stacked = self.reward_obsfeat_fn(sampbatch.obs.stacked)
				# Reward is computed wrt current reward function
				# TODO: normalize rewards
				rcurr_stacked = self.reward_func.compute_reward(samp_robsfeat_stacked, sampbatch.a.stacked, sampbatch.time.stacked)
				assert rcurr_stacked.shape == (samp_robsfeat_stacked.shape[0],)

				# If we're regularizing the policy, add negative log probabilities to the rewards
				# Intuitively, the policy gets a bonus for being less certain of its actions
				orig_rcurr_stacked = rcurr_stacked.copy()
				if self.policy_ent_reg is not None and self.policy_ent_reg != 0:
					assert self.policy_ent_reg > 0
					# XXX probably faster to compute this from sampbatch.adist instead
					actionlogprobs_B = self.policy.compute_action_logprobs(samp_pobsfeat.stacked, sampbatch.a.stacked)
					policyentbonus_B = -self.policy_ent_reg * actionlogprobs_B
					rcurr_stacked += policyentbonus_B
				else:
					policyentbonus_B = np.zeros_like(rcurr_stacked,dtype=theano.config.floatX)

				rcurr = RaggedArray(rcurr_stacked, lengths=sampbatch.r.lengths)
				# print "Returned reward shape: ", len(rcurr.arrays) 


				# Compute advantages using these rewards 
				advantages, qvals, vfunc_r2, simplev_r2 = rl.compute_advantage(
					rcurr, samp_pobsfeat, sampbatch.time, self.value_func, self.discount, self.lam)
				
				# Compute discounted costs for individual trajectories
				traj_costs = self.compute_discounted_traj_costs(qvals)
				# print "Extracted discounted trajectory costs: ", traj_costs

			# Calculating CVaR Weights
			# For policy
			policyparam_CVaR_weights = get_policyparam_CVaR_weights(traj_costs) 
			# print "len of policyparam_CVaR_weights ", len(policyparam_CVaR_weights)
			policyparam_CVaR_weights = np.repeat(policyparam_CVaR_weights, sampbatch.a.lengths) #To work with sampbatch.a.stacked 

			# print "len of policyparam_CVaR_weights ", len(policyparam_CVaR_weights)
			# print "len of sampbatch.a.stacked", len(sampbatch.a.stacked)

			# For reward
			geometric_sum_discount = [(1-self.discount**traj_len)/(1-self.discount) for traj_len in sampbatch.a.lengths]
			rewardparam_CVaR_weights = get_rewardparam_CVaR_weights(traj_costs, geometric_sum_discount) 
			rewardparam_CVaR_weights = np.repeat(rewardparam_CVaR_weights, sampbatch.a.lengths) #To work with sampbatch.a.stacked 
			# print "len of rewardparam_CVaR_weights ", len(rewardparam_CVaR_weights)

			# For CVaR params
			nu_gradient = get_CVaR_nu_gradient(traj_costs)
			Lambda_gradient = get_CVaR_Lambda_gradient(traj_costs)

			# >>>>>>>>>>>>>>>>>>> Take a step <<<<<<<<<<<<<<<<<<<<

			# print 'Fitting policy'
			with util.Timer() as t_step:

				# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Policy parameters are extracted here <<<<<<<<<<<<<<<<<<<<<<<<<<<
				params0_P = self.policy.get_params()
				# print "Num Policy params: ", len(params0_P)
				objgrad_P_test = self.policy.gradients_for_debugging(samp_pobsfeat.stacked, sampbatch.a.stacked, sampbatch.adist.stacked,
					advantages.stacked, policyparam_CVaR_weights)
				# print "Len of gradient of obj wrt policy params: ", len(objgrad_P_test)

				step_print = self.step_func(
					self.policy, params0_P,
					samp_pobsfeat.stacked, sampbatch.a.stacked, sampbatch.adist.stacked,
					advantages.stacked,
					policyparam_CVaR_weights)

				self.policy.update_obsnorm(samp_pobsfeat.stacked)

			# Fit reward function
			# print 'Fitting reward'
			with util.Timer() as t_r_fit:
				if True:#self.curr_iter % 20 == 0:
					# Subsample expert transitions to the same sample count for the policy
					inds = np.random.choice(self.ex_robsfeat.shape[0], size=samp_pobsfeat.stacked.shape[0])
					exbatch_robsfeat = self.ex_robsfeat[inds,:]
					exbatch_pobsfeat = self.ex_pobsfeat[inds,:] # only used for logging
					exbatch_a = self.ex_a[inds,:]
					exbatch_t = self.ex_t[inds]
					rfit_print = self.reward_func.fit(samp_robsfeat_stacked, sampbatch.a.stacked, sampbatch.time.stacked, exbatch_robsfeat, exbatch_a, exbatch_t, rewardparam_CVaR_weights)
				else:
					rfit_print = []

			# Fit value function for next iteration
			# print 'Fitting value function'
			with util.Timer() as t_vf_fit:
				if self.value_func is not None:
					# Recompute q vals # XXX: this is only necessary if fitting reward after policy
					# qnew = qvals

					# TODO: this should be a byproduct of reward fitting
					rnew = RaggedArray(
						self.reward_func.compute_reward(samp_robsfeat_stacked, sampbatch.a.stacked, sampbatch.time.stacked),
						lengths=sampbatch.r.lengths)
					qnew, _ = rl.compute_qvals(rnew, self.discount)
					vfit_print = self.value_func.fit(samp_pobsfeat.stacked, sampbatch.time.stacked, qnew.stacked)
				else:
					vfit_print = []

			with util.Timer() as t_CVaR_params:
				# print "CVaR params before:", self.CVaR.nu.eval(), self.CVaR.Lambda.eval()
				self.CVaR.fit([nu_gradient, Lambda_gradient], learning_rate=self.CVaR_lr)
				# print "CVaR params:", type(self.CVaR.nu.eval()) , type(self.CVaR.Lambda.eval())
		# Log
		self.total_num_trajs += len(sampbatch)
		self.total_num_sa += sum(len(traj) for traj in sampbatch)
		self.total_time += t_all.dt
		fields = [
			('iter', self.curr_iter, int),
			('trueret', sampbatch.r.padded(fill=0.).sum(axis=1).mean(), float), # average return for this batch of trajectories
			('iret', rcurr.padded(fill=0.).sum(axis=1).mean(), float),
			('trueret_std', sampbatch.r.padded(fill=0.).sum(axis=1).std(), float),
			('ire_std', rcurr.padded(fill=0.).sum(axis=1).std(), float),# std return on imitation reward
			('nu', float(self.CVaR.nu.eval()), float),
			('Lambda', float(self.CVaR.Lambda.eval()), float),#] 
			('num_traj', int(sampbatch.__len__()), int),
		#     ('avglen', int(np.mean([len(traj) for traj in sampbatch])), int), # average traj length
		#     ('ntrajs', self.total_num_trajs, int), # total number of trajs sampled over the course of training
		#     ('nsa', self.total_num_sa, int), # total number of state-action pairs sampled over the course of training
		#     ('ent', self.policy._compute_actiondist_entropy(sampbatch.adist.stacked).mean(), float), # entropy of action distributions
		#     ('vf_r2', vfunc_r2, float),
		#     ('tdvf_r2', simplev_r2, float),
		#     ('dx', util.maxnorm(params0_P - self.policy.get_params()), float), # max parameter difference from last iteration
		# ] + step_print + vfit_print + rfit_print + [
		#     ('avgr', rcurr_stacked.mean(), float), # average regularized reward encountered
		#     ('avgunregr', orig_rcurr_stacked.mean(), float), # average unregularized reward
		#     ('avgpreg', policyentbonus_B.mean(), float), # average policy regularization
		#     # ('bcloss', -self.policy.compute_action_logprobs(exbatch_pobsfeat, exbatch_a).mean(), float), # negative log likelihood of expert actions
		#     # ('bcloss', np.square(self.policy.compute_actiondist_mean(exbatch_pobsfeat) - exbatch_a).sum(axis=1).mean(axis=0), float),
		#    ('tsamp', t_sample.dt, float), # time for sampling
		#    ('tadv', t_adv.dt + t_vf_fit.dt, float), # time for advantage computation
		#    ('tstep', t_step.dt, float), # time for step computation
		#    ('t_r_fit', t_r_fit.dt, float), # time for reward fitting
		#    ('t_vf_fit', t_vf_fit.dt, float), # time for value-function fitting
		#    ('t_all', t_all.dt, float)
			# ('ttotal', self.total_time, float), # total time
		]
		fields.extend(step_print)
		fields.extend(rfit_print)
		self.curr_iter += 1
		return fields

	def compute_discounted_traj_costs(self, qvals):
		"""qvals is a RaggedArray with the q values of len(qvars.arrays) trajectories """
		traj_costs = []
		for i in range(len(qvals.arrays)):#For each trajectory
		# ============================================================================================
			traj_costs.append(-qvals[i][0]) # !!!! We need costs and not rewards !!!!
		# ============================================================================================
		return traj_costs 

