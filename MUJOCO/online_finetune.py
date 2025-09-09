import numpy as np
import torch
#import gym
import argparse
import os
import yaml
from tools4online import TD3, New_Trans_RB, env_constructor
import copy
#########################################################
from collections import defaultdict


########################################################

from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"


evaluations = []


def eval_transformer_2stage(policy, args, eval_episodes=10):
    
    eval_env, state_dim, action_dim = env_constructor(args.env, seed=args.seed+100, obs_indices=args.obs_indices)
    avg_reward = 0.
    policy.trans.eval()

    for _ in range(eval_episodes):
        
        #eval_env.seed(args.seed+100)
        state = eval_env.reset()
        done = False
        
        st_state = state#['state']            #state (n_e, s_d)
        st_states = copy.deepcopy(st_state).unsqueeze(1).to(device=device, dtype=torch.float32)  #state (n_e, 1, s_d)
        if args.obs_mode != 'state':
            img_state = state[f"{args.obs_mode}"]   #img_state (n_e, 128, 128, 4/3)
            img_states = copy.deepcopy(img_state).unsqueeze(1).to(device=device, dtype=torch.float32)  #img_states (n_e, 1, 128, 128, 4/3)
        
        t = -1
        while done == False:
            t+=1
            if st_states.shape[1] > policy.context_length : 
                st_states = st_states[:, -policy.context_length:, :]
                if args.obs_mode != 'state':
                    img_states = img_states[:, -policy.context_length:, :]

            st_s = st_states.unsqueeze(1)             #st_s (n_e, 1, cont, s_d)
            if args.obs_mode != 'state':
                img_s = img_states.unsqueeze(1)              #img_s (n_e, 1, cont, 128, 128, 4/3)
            sampled_action = policy.trans.actor_forward(st_s) if args.obs_mode == 'state' else policy.trans.actor_forward(st_s, img_s)  # sampled_action (n_e, 1, a_d)
            sampled_action = sampled_action.detach().cpu()
            action = np.clip( sampled_action.numpy()[:,0,], -1, 1)  #action (n_e, a_d)

            
            state, r, done, info = eval_env.step( action[0] )
            
            avg_reward += r.item()
            st_state = state#['state'] 
            st_cur_state = copy.deepcopy(st_state).unsqueeze(1).to(device=device, dtype=torch.float32)  #cur_state (n_e, 1, s_d)
            st_states = torch.cat([st_states, st_cur_state], dim=1)                                          #states (n_e, cont+1, s_d)
            if args.obs_mode != 'state':
                img_state = state[f"{args.obs_mode}"]   #img_state (n_e, 128, 128, 4/3)
                img_cur_state = copy.deepcopy(img_state).unsqueeze(1).to(device=device, dtype=torch.float32)  #img_cur_state (n_e, 1, 128, 128, 4/3)
                img_states = torch.cat([img_states, img_cur_state], dim=1)                                          #img_states (n_e, cont, 128, 128, 4/3)

        
            
            

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    
    policy.trans.train()
    
    return avg_reward


def second_stage(policy, config, args, experiment):
  

    env, state_dim, action_dim = env_constructor(args.env, seed=args.seed, obs_indices=args.obs_indices)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    
    context_length = config['train_config']['context_length']

    policy.new_trans_RB = New_Trans_RB(args.num_envs, config['train_config']['replay_buffer_size'], context_length, state_dim, action_dim, args.obs_mode)
    policy.experiment = experiment
    
    eval_counter = 0
    eval_counter += 1
    avg_reward = eval_transformer_2stage(policy, args, 20)
    experiment.add_scalar('Eval_reward', avg_reward, 1)
    
    
    policy.trans.train()
    #policy.critic.train()

    out_states = []
    if args.obs_mode != 'state':
        out_img_states = []
    out_actions = []
    out_rewards = []
    out_dones = []
    
    
    state = env.reset()
    
    st_state = state#['state']
    if args.obs_mode != 'state':
        img_state = state[f"{args.obs_mode}"]
    
    out_states.append( st_state )
    if args.obs_mode != 'state':
        out_img_states.append(img_state)
    
    st_states = copy.deepcopy(st_state).unsqueeze(1).to(device=device, dtype=torch.float32)  #state (n_e, 1, s_d)

    if args.obs_mode != 'state':
        img_states = copy.deepcopy(img_state).unsqueeze(1).to(device=device, dtype=torch.float32)
    
    avg_ret = 0.
    avg_reward = 0.
    episode_num = 0
    
    done = False
    avg_reward = eval_transformer_2stage(policy, args, 10)
    experiment.add_scalar('Eval_reward', avg_reward, 0)
    for t in range(int(args.max_timesteps)):

        if st_states.shape[1] > context_length : 
            st_states = st_states[:, -context_length:, :]
            if args.obs_mode != 'state':
                img_states = img_states[:, -context_length:, :]    

        st_s = st_states.unsqueeze(1)    # n_e, 1, cont, s_d
        if args.obs_mode != 'state':
            img_s = img_states.unsqueeze(1) 
        
        if t < args.start_timesteps:
            action = env.action_space.sample()  #sampled_action=arr(n_e, a_d)
            #action = np.expand_dims(action, axis=0)
            out_actions.append(torch.Tensor(action).unsqueeze(0))
        else:
            
            sampled_action = policy.trans.actor_forward(st_s)
            sampled_action = sampled_action.detach().cpu()[:,0,]            #sampled_action=Tensor(n_e, a_d)
            out_actions.append(sampled_action)
            action = np.clip( sampled_action.numpy()[0], -1, 1)                #action=Arr(n_e, a_d) 

        
        state, reward, done, info = env.step( action )
        avg_ret += reward.item()

        
        out_dones.append(done.to(float).reshape(-1, 1))    #truncated.reshape(-1, 1) = (n_e, 1)
        out_rewards.append(reward.reshape(-1, 1))               #reward.reshape(-1, 1) = (n_e, 1)
        
        st_state = state#['state']
        out_states.append( state ) #state['state']
        if args.obs_mode != 'state':
            img_state = state[f"{args.obs_mode}"]
            out_img_states.append(img_state)

        if t >= context_length-1 and st_states.shape[1] == context_length:
            states2RB = out_states[-context_length-1:-1]
            if args.obs_mode != 'state':
                img_states2RB = out_img_states[-context_length-1:-1]
                img_next_states2RB = out_img_states[-context_length:]
            act2RB = out_actions[-1]
            ret2RB = out_rewards[-1]
            done2RB = out_dones[-1]
            next_states2RB = out_states[-context_length:]
            
            if args.obs_mode == 'state':
                policy.new_trans_RB.recieve_traj(states2RB, act2RB, ret2RB, done2RB, next_states2RB)
            else:    
                policy.new_trans_RB.recieve_traj(states2RB, act2RB, ret2RB, done2RB, next_states2RB, img_states2RB, img_next_states2RB)

        st_cur_state = copy.deepcopy(st_state).unsqueeze(1).to(device=device, dtype=torch.float32)  # (n_e, 1, s_d)
        st_states = torch.cat([st_states, st_cur_state], dim=1)
        if args.obs_mode != 'state':
            img_cur_state = copy.deepcopy(img_state).unsqueeze(1).to(device=device, dtype=torch.float32)  #img_cur_state (n_e, 1, 128, 128, 4/3)
            img_states = torch.cat([img_states, img_cur_state], dim=1)   
        
        
        if t >= args.start_timesteps: 
            policy.stage_2_train(args.batch_size)
               

        if done:
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Reward: {avg_ret:.3f}")
            avg_ret = 0
            
            out_states = []
            if args.obs_mode != 'state':
                out_img_states = []
            out_actions = []
            out_rewards = []
            out_dones = []
            done = False
            
            #env.seed(args.seed)
            state = env.reset()

            st_state = state#['state']
            if args.obs_mode != 'state':
                img_state = state[f"{args.obs_mode}"]
            
            out_states.append( st_state )
            if args.obs_mode != 'state':
                out_img_states.append(img_state)
            
            st_states = copy.deepcopy(st_state).unsqueeze(1).to(device=device, dtype=torch.float32)  #state (n_e, 1, s_d)
            if args.obs_mode != 'state':
                img_states = copy.deepcopy(img_state).unsqueeze(1).to(device=device, dtype=torch.float32)

            episode_num += 1


        # Evaluate episode
        if ((t + 1) % args.eval_freq == 0) and (t >= args.start_timesteps):
            #eval_counter += 1
            avg_reward = eval_transformer_2stage(policy, args, 10)
            experiment.add_scalar('Eval_reward', avg_reward, t)
    
    torch.save(policy.trans, f"WORKSHOP_WEIGHTS/{args.env}/[fine_tuned]Trans|seed={args.seed}|AddAsc={args.additional_ascent}|UseTrData={args.use_train_data}|.pth")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="HalfCheetah-v4")          # OpenAI gym environment name
    parser.add_argument("--obs_indices", default=None) #Cth [0,1,2,3,8,9,10,11,12] | Hppr [0,1,2,3,4] | Ant [0,1,2,3,4,5,6,7,8,9,10,11,12]
    parser.add_argument("--obs_mode", default="state") 
    parser.add_argument("--use_train_data", default=False)
    parser.add_argument("--additional_ascent", default=True)
    parser.add_argument("--num_envs", default=1, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--trans_critic", default=False)
    parser.add_argument("--separate_trans_critic", default=False)
    parser.add_argument("--start_timesteps", default=0, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=3e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=500000, type=int)   # Max time steps to run environment
    parser.add_argument("--grad_clip", default=100000, type=int)
    parser.add_argument("--expl_noise", default=0.1, type=float)    #  0.0
    parser.add_argument("--batch_size", default=256, type=int)      # 256
    parser.add_argument("--discount", default=0.99, type=float)     # 0.8
    parser.add_argument("--tau", default=0.005, type=float)         # 0.005
    parser.add_argument("--policy_noise", default=0.2)              #  0.1
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name   
    args = parser.parse_args()
    
    with open("MUJOCO/sh_config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    
    
    for RUN in [2,4]:
        args.seed = RUN
        path2run = f"WORKSHOP_RUNS/{args.env}/[online_finetune]|seed={args.seed}|AddAsc={args.additional_ascent}|UseTrData={args.use_train_data}|"
    
        experiment = SummaryWriter(log_dir=path2run)
    
        test_env, state_dim, action_dim = env_constructor(args.env, seed=args.seed)
    
        kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": 1,
            "discount": args.discount,
            "tau": args.tau,
        }
    
        # Initialize policy
        if args.policy == "TD3":
            # Target policy smoothing is scaled wrt the action scale
            kwargs["policy_noise"] = args.policy_noise * 1
            kwargs["noise_clip"] = args.noise_clip * 1
            kwargs["policy_freq"] = args.policy_freq
            kwargs["grad_clip"] = args.grad_clip

        #140 
        pth2trans = f"WORKSHOP_WEIGHTS/{args.env}/[accelerated]Trans|seed={args.seed}|AddAsc={args.additional_ascent}|UseTrData={args.use_train_data}|.pth"
        pth2trans_tgt = f"WORKSHOP_WEIGHTS/{args.env}/[accelerated]Trans(t)|seed={args.seed}|AddAsc={args.additional_ascent}|UseTrData={args.use_train_data}|.pth"
        pth2critic = f"WORKSHOP_WEIGHTS/{args.env}/[accelerated]St_Critic|seed={args.seed}|AddAsc={args.additional_ascent}|UseTrData={args.use_train_data}|.pth"
        pth2critic_tgt = f"WORKSHOP_WEIGHTS/{args.env}/[accelerated]St_Critic(t)|seed={args.seed}|AddAsc={args.additional_ascent}|UseTrData={args.use_train_data}|.pth"
        
        kwargs["preload_weights"] = [pth2trans, pth2trans_tgt, pth2critic, pth2critic_tgt]
        policy = TD3(args.num_envs, 'state', config['train_config']['context_length'], config['model_config'], **kwargs)
        
        second_stage(policy, config, args, experiment)                    