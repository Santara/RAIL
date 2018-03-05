import argparse, h5py, json
import numpy as np
from environments import rlgymenv
import policyopt
from policyopt import imitation, nn, rl, util
import pdb
import sys
import theano
import pickle as pkl
sys.setrecursionlimit(100000000)

MODES = ('bclone', 'ga')
OBSNORM_MODES = ('none', 'expertdata', 'online')
TINY_ARCHITECTURE = '[{"type": "fc", "n": 64}, {"type": "nonlin", "func": "tanh"}, {"type": "fc", "n": 64}, {"type": "nonlin", "func": "tanh"}]'
SIMPLE_ARCHITECTURE = '[{"type": "fc", "n": 100}, {"type": "nonlin", "func": "tanh"}, {"type": "fc", "n": 100}, {"type": "nonlin", "func": "tanh"}]'


def load_dataset(filename, limit_trajs, data_subsamp_freq, len_filtering, len_filter_threshold):
    # Load expert data
    with h5py.File(filename, 'r') as f:
        # Read data as written by vis_mj.py
        full_dset_size = f['obs_B_T_Do'].shape[0] # full dataset size
        dset_size = min(full_dset_size, limit_trajs) if limit_trajs is not None else full_dset_size

        exobs_B_T_Do = f['obs_B_T_Do'][:dset_size,...][...]
        exa_B_T_Da = f['a_B_T_Da'][:dset_size,...][...]
        exr_B_T = f['r_B_T'][:dset_size,...][...] #reward obtained at each step in each trajectory
        exlen_B = f['len_B'][:dset_size,...][...] #Length of trajectories

    # Filter trajectories by length: include only the top 80-percentile : experimental - by anirban
    # pdb.set_trace()
    if len_filtering:
        n_trajs = exobs_B_T_Do.shape[0]
        len_thresh = np.percentile(exlen_B,len_filter_threshold)
        filtered_indices = exlen_B>=len_thresh

        exobs_B_T_Do = exobs_B_T_Do[filtered_indices]
        exa_B_T_Da = exa_B_T_Da[filtered_indices]
        exr_B_T = exr_B_T[filtered_indices]
        exlen_B = exlen_B[filtered_indices]

        print "%d trajectories removed" %(n_trajs-exobs_B_T_Do.shape[0])




    print 'Expert dataset size: {} transitions ({} trajectories)'.format(exlen_B.sum(), len(exlen_B))
    print 'Expert average return:', exr_B_T.sum(axis=1).mean()

    # Stack everything together
    start_times_B = np.random.RandomState(0).randint(0, data_subsamp_freq, size=exlen_B.shape[0]) 
    print 'start times'
    print start_times_B
    exobs_Bstacked_Do = np.concatenate(
        [exobs_B_T_Do[i,  start_times_B[i] : l : data_subsamp_freq,  :] for i, l in enumerate(exlen_B)],
        axis=0)
    exa_Bstacked_Da = np.concatenate(
        [exa_B_T_Da[i,start_times_B[i]:l:data_subsamp_freq,:] for i, l in enumerate(exlen_B)],
        axis=0)
    ext_Bstacked = np.concatenate(
        [np.arange(start_times_B[i], l, step=data_subsamp_freq) for i, l in enumerate(exlen_B)]).astype(theano.config.floatX)

    assert exobs_Bstacked_Do.shape[0] == exa_Bstacked_Da.shape[0] == ext_Bstacked.shape[0]# == np.ceil(exlen_B.astype(float)/data_subsamp_freq).astype(int).sum() > 0

    print 'Subsampled data every {} timestep(s)'.format(data_subsamp_freq)
    print 'Final dataset size: {} transitions (average {} per traj)'.format(exobs_Bstacked_Do.shape[0], float(exobs_Bstacked_Do.shape[0])/dset_size)

    return exobs_Bstacked_Do, exa_Bstacked_Da, ext_Bstacked


def main():
    np.set_printoptions(suppress=True, precision=5, linewidth=1000)

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=MODES, required=True)
    # Expert dataset
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--resume_training', action='store_true', help="Resume training from a checkpoint: --policy_checkpoint. Currently only supports GAIL with nn policy, reward and vf") 
    parser.add_argument('--checkpoint', type=str, help="Load from checkpoint if provided and if --resume_training") 
    parser.add_argument('--limit_trajs', type=int, required=True, help="How many expert trajectories to be used for training. If None : full dataset is used.") 
    parser.add_argument('--data_subsamp_freq', type=int, required=True, help="A number between 0 and max_traj_len. Rate of subsampling of expert trajectories while creating the dataset of expert transitions (state-action)")
    # MDP options
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--max_traj_len', type=int, default=None)
    # Policy architecture
    parser.add_argument('--policy_hidden_spec', type=str, default=SIMPLE_ARCHITECTURE)
    parser.add_argument('--tiny_policy', action='store_true')
    parser.add_argument('--obsnorm_mode', choices=OBSNORM_MODES, default='expertdata')
    # Behavioral cloning optimizer
    parser.add_argument('--bclone_lr', type=float, default=1e-3)
    parser.add_argument('--bclone_batch_size', type=int, default=128)
    # parser.add_argument('--bclone_eval_nsa', type=int, default=128*100)
    parser.add_argument('--bclone_eval_ntrajs', type=int, default=20)
    parser.add_argument('--bclone_eval_freq', type=int, default=1000)
    parser.add_argument('--bclone_train_frac', type=float, default=.7)
    # Imitation optimizer
    parser.add_argument('--discount', type=float, default=.995)

    parser.add_argument('--lam', type=float, default=.97)
    parser.add_argument('--max_iter', type=int, default=1000000)
    parser.add_argument('--policy_max_kl', type=float, default=.01)
    parser.add_argument('--policy_cg_damping', type=float, default=.1, help="TRPO parameter")
    parser.add_argument('--no_vf', type=int, default=0)
    parser.add_argument('--vf_max_kl', type=float, default=.01)
    parser.add_argument('--vf_cg_damping', type=float, default=.1)
    parser.add_argument('--policy_ent_reg', type=float, default=0.)
    parser.add_argument('--reward_type', type=str, default='nn')
    # parser.add_argument('--linear_reward_bin_features', type=int, default=0)
    parser.add_argument('--reward_max_kl', type=float, default=.01, help="TRPO parameter") 
    parser.add_argument('--reward_lr', type=float, default=.01)
    parser.add_argument('--reward_steps', type=int, default=1)
    parser.add_argument('--reward_ent_reg_weight', type=float, default=.001)
    parser.add_argument('--reward_include_time', type=int, default=0)
    parser.add_argument('--sim_batch_size', type=int, default=None)
    parser.add_argument('--min_total_sa', type=int, default=50000)
    parser.add_argument('--favor_zero_expert_reward', type=int, default=0)
    # Saving stuff
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=20)
    parser.add_argument('--plot_freq', type=int, default=0)
    parser.add_argument('--log', type=str, required=False)
    # CVaR parameters
    parser.add_argument('--useCVaR', action='store_true')
    parser.add_argument('--CVaR_alpha', type=float, default=0.9)
    parser.add_argument('--CVaR_beta', type=float, default=0.)
    parser.add_argument('--CVaR_lr', type=float, default=0.01)
    # !!! The following argument --disc_CVaR_weight is not of use and should be removed
    parser.add_argument('--disc_CVaR_weight', type=float, default=1., help="Weight given to CVaR loss for the discriminator. Added by Anirban for smooth convergence.")
    parser.add_argument('--CVaR_Lambda_not_trainable', action='store_false')
    parser.add_argument('--CVaR_Lambda_val_if_not_trainable', type=float, default=0.5)
    #Filtering expert trajectories
    parser.add_argument('--use_expert_traj_filtering', action='store_true')
    parser.add_argument('--expert_traj_filt_percentile_threshold', type=float, default=20)
    # Additive state prior formulation
    parser.add_argument('--use_additiveStatePrior', action='store_true')
    parser.add_argument('--additiveStatePrior_weight', type=float, default=1.)
    parser.add_argument('--n_gmm_components', type=int, default=5)
    parser.add_argument('--cov_type_gmm', type=str, default='diag')
    parser.add_argument('--familiarity_alpha', type=float, default=10000000)
    parser.add_argument('--familiarity_beta', type=float, default=100)

    parser.add_argument('--kickThreshold_percentile', type=float, default=100.0)
    parser.add_argument('--appendFlag', action='store_true')

    args = parser.parse_args()

    if args.useCVaR:
        print ">>>>>>>>>>>>>>>>>>> TRAINING RAIL <<<<<<<<<<<<<<<<<<<"
    elif args.use_additiveStatePrior:
        print ">>>>>>>>>>>>>>>>>>> USING ADDITIVE STATE PRIOR <<<<<<<<<<<<<<<<<<<"
    else:
    	print ">>>>>>>>> TRAINING GAIL <<<<<<<<<<"

    # Initialize the MDP
    if args.tiny_policy:
        assert args.policy_hidden_spec == SIMPLE_ARCHITECTURE, 'policy_hidden_spec must remain unspecified if --tiny_policy is set'
        args.policy_hidden_spec = TINY_ARCHITECTURE
    argstr = json.dumps(vars(args), separators=(',', ':'), indent=2)
    print(argstr)
    print "\n\n========== Policy network specifications loaded ===========\n\n"

    mdp = rlgymenv.RLGymMDP(args.env_name)
    util.header('MDP observation space, action space sizes: %d, %d\n' % (mdp.obs_space.dim, mdp.action_space.storage_size))

    print "\n\n========== MDP initialized ===========\n\n"

    # Initialize the policy
    enable_obsnorm = args.obsnorm_mode != 'none'
    if isinstance(mdp.action_space, policyopt.ContinuousSpace):
        policy_cfg = rl.GaussianPolicyConfig(
            hidden_spec=args.policy_hidden_spec,
            min_stdev=0.,
            init_logstdev=0.,
            enable_obsnorm=enable_obsnorm)
        policy = rl.GaussianPolicy(policy_cfg, mdp.obs_space, mdp.action_space, 'GaussianPolicy', args.useCVaR)
    else:
        policy_cfg = rl.GibbsPolicyConfig(
            hidden_spec=args.policy_hidden_spec,
            enable_obsnorm=enable_obsnorm)
        policy = rl.GibbsPolicy(policy_cfg, mdp.obs_space, mdp.action_space, 'GibbsPolicy', args.useCVaR)

    offset = 0
    #Load from checkpoint if provided <<<<<<<<<<<<<=============================>>>>>>>>>>>>>>>>.
    if args.resume_training:
        if args.checkpoint is not None:
            file, policy_key = util.split_h5_name(args.checkpoint)
            offset = int(policy_key.split('/')[-1][4:])
            print '\n**************************************************'
            print 'Resuming from checkpoint : %d of %s' % (offset, file)
            print  '**************************************************\n'

            if args.appendFlag and file!=args.log:
                raise RuntimeError('Log file and checkpoint should have the same name if appendFlag is on. %s vs %s' % file, args.log)

            policy_file = file[:-3]+'_policy.h5'                  # Because we're naming the file as *_policy.h5 itself
            policy.load_h5(policy_file, policy_key)

    util.header('Policy architecture')
    for v in policy.get_trainable_variables():
        util.header('- %s (%d parameters)' % (v.name, v.get_value().size))
    util.header('Total: %d parameters' % (policy.get_num_params(),))

    print "\n\n========== Policy initialized ===========\n\n"

    # Load expert data

    exobs_Bstacked_Do, exa_Bstacked_Da, ext_Bstacked = load_dataset(
        args.data, args.limit_trajs, args.data_subsamp_freq, 
        len_filtering=args.use_expert_traj_filtering, 
        len_filter_threshold=args.expert_traj_filt_percentile_threshold)


    assert exobs_Bstacked_Do.shape[1] == mdp.obs_space.storage_size
    assert exa_Bstacked_Da.shape[1] == mdp.action_space.storage_size
    assert ext_Bstacked.ndim == 1

    print "\n\n========== Expert data loaded ===========\n\n"

    print '\n==================== Hyperparams ===================='
    print '\texpert_traj_filt_percentile_threshold = %f' % args.expert_traj_filt_percentile_threshold
    print '\tfamiliarity_alpha = %f' % args.familiarity_alpha
    print '\tfamiliarity_beta = %f' % args.familiarity_beta
    print '\tkickThreshold_percentile = %f' % args.kickThreshold_percentile
    print '==============================================\n'

    # Start optimization
    max_traj_len = args.max_traj_len if args.max_traj_len is not None else mdp.env_spec.timestep_limit
    print 'Max traj len:', max_traj_len

    if args.mode == 'bclone':
        # For behavioral cloning, only print output when evaluating
        args.print_freq = args.bclone_eval_freq
        args.save_freq = args.bclone_eval_freq

        reward, vf = None, None #There is no role of the reward function or value function in behavioral cloning
        opt = imitation.BehavioralCloningOptimizer(
            mdp, policy,
            lr=args.bclone_lr,
            batch_size=args.bclone_batch_size,
            obsfeat_fn=lambda o:o,
            ex_obs=exobs_Bstacked_Do, ex_a=exa_Bstacked_Da,
            eval_sim_cfg=policyopt.SimConfig(
                min_num_trajs=args.bclone_eval_ntrajs, min_total_sa=-1,
                batch_size=args.sim_batch_size, max_traj_len=max_traj_len),
            eval_freq=args.bclone_eval_freq,
            train_frac=args.bclone_train_frac)

        print "======= Behavioral Cloning optimizer initialized ======="

    elif args.mode == 'ga':
        if args.reward_type == 'nn':
            reward = imitation.TransitionClassifier( #Add resume training functionality
                hidden_spec=args.policy_hidden_spec,
                obsfeat_space=mdp.obs_space,
                action_space=mdp.action_space,
                max_kl=args.reward_max_kl,
                adam_lr=args.reward_lr,
                adam_steps=args.reward_steps,
                ent_reg_weight=args.reward_ent_reg_weight,
                enable_inputnorm=True,
                include_time=bool(args.reward_include_time),
                time_scale=1./mdp.env_spec.timestep_limit,
                favor_zero_expert_reward=bool(args.favor_zero_expert_reward),
                varscope_name='TransitionClassifier',
                useCVaR = args.useCVaR,
                CVaR_loss_weightage = args.disc_CVaR_weight)
            #Load from checkpoint if provided <<<<<<<<<<<<<=============================>>>>>>>>>>>>>>>>.
            if args.resume_training:
                if args.checkpoint is not None:
                    file, reward_key = util.split_h5_name(args.checkpoint)
                    reward_file = file[:-3]+'_reward.h5'
                    print reward_file
                    reward.load_h5(reward_file, reward_key)

        elif args.reward_type in ['l2ball', 'simplex']:
            reward = imitation.LinearReward(
                obsfeat_space=mdp.obs_space,
                action_space=mdp.action_space,
                mode=args.reward_type,
                enable_inputnorm=True,
                favor_zero_expert_reward=bool(args.favor_zero_expert_reward),
                include_time=bool(args.reward_include_time),
                time_scale=1./mdp.env_spec.timestep_limit,
                exobs_Bex_Do=exobs_Bstacked_Do,
                exa_Bex_Da=exa_Bstacked_Da,
                ext_Bex=ext_Bstacked
                )
        else:
            raise NotImplementedError(args.reward_type)

        vf = None if bool(args.no_vf) else rl.ValueFunc( #Add resume training functionality
            hidden_spec=args.policy_hidden_spec,
            obsfeat_space=mdp.obs_space,
            enable_obsnorm=args.obsnorm_mode != 'none',
            enable_vnorm=True,
            max_kl=args.vf_max_kl,
            damping=args.vf_cg_damping,
            time_scale=1./mdp.env_spec.timestep_limit,
            varscope_name='ValueFunc')
        if args.resume_training:
            if args.checkpoint is not None:
                file, vf_key = util.split_h5_name(args.checkpoint)
                vf_file = file[:-3]+'_vf.h5'
                vf.load_h5(vf_file, vf_key)
        if args.useCVaR:
            opt = imitation.ImitationOptimizer_CVaR(
                mdp=mdp,
                discount=args.discount,
                lam=args.lam,
                policy=policy,
                sim_cfg=policyopt.SimConfig(
                    min_num_trajs=-1, min_total_sa=args.min_total_sa,
                    batch_size=args.sim_batch_size, max_traj_len=max_traj_len),
                step_func=rl.TRPO(max_kl=args.policy_max_kl, damping=args.policy_cg_damping, useCVaR=True),
                reward_func=reward,
                value_func=vf,
                policy_obsfeat_fn=lambda obs: obs,
                reward_obsfeat_fn=lambda obs: obs,
                policy_ent_reg=args.policy_ent_reg,
                ex_obs=exobs_Bstacked_Do,
                ex_a=exa_Bstacked_Da,
                ex_t=ext_Bstacked,
                #For CVaR
                CVaR_alpha=args.CVaR_alpha,
                CVaR_beta=args.CVaR_beta,
                CVaR_lr = args.CVaR_lr,
                CVaR_Lambda_trainable = args.CVaR_Lambda_not_trainable,
                CVaR_Lambda_val_if_not_trainable = args.CVaR_Lambda_val_if_not_trainable,
                offset=offset+1)
        elif args.use_additiveStatePrior:
            opt = imitation.ImitationOptimizer_additiveStatePrior(
            mdp=mdp,
            discount=args.discount,
            lam=args.lam,
            policy=policy,
            sim_cfg=policyopt.SimConfig(
                min_num_trajs=-1, min_total_sa=args.min_total_sa,
                batch_size=args.sim_batch_size, max_traj_len=max_traj_len),
            step_func=rl.TRPO(max_kl=args.policy_max_kl, damping=args.policy_cg_damping, useCVaR=False),
            reward_func=reward,
            value_func=vf,
            policy_obsfeat_fn=lambda obs: obs,
            reward_obsfeat_fn=lambda obs: obs,
            policy_ent_reg=args.policy_ent_reg,
            ex_obs=exobs_Bstacked_Do,
            ex_a=exa_Bstacked_Da,
            ex_t=ext_Bstacked,
            n_gmm_components=args.n_gmm_components, 
            cov_type_gmm=args.cov_type_gmm,
            additiveStatePrior_weight=args.additiveStatePrior_weight,
            alpha=args.familiarity_alpha,
            beta=args.familiarity_beta,
            kickThreshold_percentile=args.kickThreshold_percentile, 
            offset=offset+1)
        else:
            opt = imitation.ImitationOptimizer(
            mdp=mdp,
            discount=args.discount,
            lam=args.lam,
            policy=policy,
            sim_cfg=policyopt.SimConfig(
                min_num_trajs=-1, min_total_sa=args.min_total_sa,
                batch_size=args.sim_batch_size, max_traj_len=max_traj_len),
            step_func=rl.TRPO(max_kl=args.policy_max_kl, damping=args.policy_cg_damping, useCVaR=False),
            reward_func=reward,
            value_func=vf,
            policy_obsfeat_fn=lambda obs: obs,
            reward_obsfeat_fn=lambda obs: obs,
            policy_ent_reg=args.policy_ent_reg,
            ex_obs=exobs_Bstacked_Do,
            ex_a=exa_Bstacked_Da,
            ex_t=ext_Bstacked)

    # Set observation normalization
    if args.obsnorm_mode == 'expertdata':
        policy.update_obsnorm(exobs_Bstacked_Do)
        if reward is not None: reward.update_inputnorm(opt.reward_obsfeat_fn(exobs_Bstacked_Do), exa_Bstacked_Da)
        if vf is not None: vf.update_obsnorm(opt.policy_obsfeat_fn(exobs_Bstacked_Do))

        print "======== Observation normalization done ========"

    # Run optimizer
    print "======== Optimization begins ========"

    # Trial: make checkpoints for policy, reward and vf
    policy_log = nn.TrainingLog(args.log[:-3]+'_policy.h5', [('args', argstr)], args.appendFlag)
    reward_log = nn.TrainingLog(args.log[:-3]+'_reward.h5', [('args', argstr)], args.appendFlag)
    vf_log = nn.TrainingLog(args.log[:-3]+'_vf.h5', [('args', argstr)], args.appendFlag)
    

    kickStatesData = []
    
    print '\n**************************************'
    print 'Running iterations from %d to %d' % (offset+1, args.max_iter)

    for i in xrange(offset+1, args.max_iter):
    # for i in range(1): #FIXME: this is just for studying the insides of the training algo
        
        # All training a.k.a. optimization happens in the next line!!! -_-
        # pdb.set_trace()
        iter_info = opt.step(i, kickStatesData) if args.use_additiveStatePrior else opt.step(i) 

        #========= The rest is fluff =============

        #Log and plot
        #pdb.set_trace()
        policy_log.write(iter_info, 
                print_header=i % (20*args.print_freq) == 0, 
                # display=False
                display=i % args.print_freq == 0 ## FIXME: AS remove comment
                )
        # reward_log.write(iter_info, 
        #         print_header=i % (20*args.print_freq) == 0, 
        #         display=False
        #         # display=i % args.print_freq == 0 ## FIXME: AS remove comment
        #         )
        # vf_log.write(iter_info, 
        #         print_header=i % (20*args.print_freq) == 0, 
        #         display=False
        #         # display=i % args.print_freq == 0 ## FIXME: AS remove comment
        #         )
        
        #FIXME: problem running this on 211 and 138. No problem on 151
        if args.save_freq != 0 and i % args.save_freq == 0 and args.log is not None:
            policy_log.write_snapshot(policy, i)
            reward_log.write_snapshot(reward, i)
            vf_log.write_snapshot(vf, i)
            
            # analysisFile=open(args.log[:-3]+'_kickedStates' + str(i) + '.pkl', 'wb')
            analysisFile=open(args.log[:-3]+'_kickedStates.pkl', 'wb')
            pkl.dump({'kickStatesData':kickStatesData}, analysisFile, protocol=2)
            analysisFile.close()

        if args.plot_freq != 0 and i % args.plot_freq == 0:
            exdata_N_Doa = np.concatenate([exobs_Bstacked_Do, exa_Bstacked_Da], axis=1)
            pdata_M_Doa = np.concatenate([opt.last_sampbatch.obs.stacked, opt.last_sampbatch.a.stacked], axis=1)

            # Plot reward
            import matplotlib.pyplot as plt
            _, ax = plt.subplots()
            idx1, idx2 = 0,1
            range1 = (min(exdata_N_Doa[:,idx1].min(), pdata_M_Doa[:,idx1].min()), max(exdata_N_Doa[:,idx1].max(), pdata_M_Doa[:,idx1].max()))
            range2 = (min(exdata_N_Doa[:,idx2].min(), pdata_M_Doa[:,idx2].min()), max(exdata_N_Doa[:,idx2].max(), pdata_M_Doa[:,idx2].max()))
            reward.plot(ax, idx1, idx2, range1, range2, n=100)

            # Plot expert data
            ax.scatter(exdata_N_Doa[:,idx1], exdata_N_Doa[:,idx2], color='blue', s=1, label='expert')

            # Plot policy samples
            ax.scatter(pdata_M_Doa[:,idx1], pdata_M_Doa[:,idx2], color='red', s=1, label='apprentice')

            ax.legend()
            plt.show()


if __name__ == '__main__':
    main()
