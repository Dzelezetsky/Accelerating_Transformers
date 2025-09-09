import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Model.transformer import Model
from gymnasium import spaces
#import gymnasium_robotics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##################################
import gym
#import gymnasium as gym
##################################


class PartialObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, obs_indices: list):
        super().__init__(env)
        self.obs_indices = np.array(obs_indices, dtype=int)
        obsspace = env.observation_space

        
        low = obsspace.low[self.obs_indices]
        high = obsspace.high[self.obs_indices]

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

    def observation(self, observation):
        
        return observation[self.obs_indices].astype(np.float32)

class GPUObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, device: torch.device):
        super().__init__(env)
        self.device = device  

    def reset(self, **kwargs):
        
        obs = self.env.reset(**kwargs)
        obs_tensor = self._to_tensor(obs).unsqueeze(0)
        #obs_tensor = self._to_tensor(obs[0]).unsqueeze(0)
        
        return obs_tensor
        
    def step(self, action):
        
        obs, reward, done, info = self.env.step(action)
        
        
        obs_tensor = self._to_tensor(obs).unsqueeze(0)
        reward_tensor = self._to_tensor(np.array(reward, dtype=np.float32))
        done_tensor = self._to_tensor(np.array(done, dtype=np.bool_))
        
        #return {'state': obs_tensor}, reward_tensor, done_tensor, info
        return obs_tensor, reward_tensor, done_tensor, info

    def _to_tensor(self, obs: np.ndarray):
        if isinstance(obs, np.ndarray):
            if np.issubdtype(obs.dtype, np.bool_):
                tensor = torch.from_numpy(obs).to(torch.bool)
            elif np.issubdtype(obs.dtype, np.floating):
                tensor = torch.from_numpy(obs).float()
            elif np.issubdtype(obs.dtype, np.integer):
                tensor = torch.from_numpy(obs).long()
            else:
                
                tensor = torch.tensor(obs, dtype=torch.float32)
            return tensor.to(self.device)
        return obs

    def seed(self, seed: int = None):
        
        return self.env.seed(seed)


def env_constructor(env_name: str, seed: int = 1, obs_indices: list = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(env_name)
    env.seed(seed)  
    
    if obs_indices is not None:
        env = PartialObservation(env, obs_indices)
    
    env = GPUObservationWrapper(env, device)
 
    return env, env.observation_space.shape[-1], env.action_space.shape[-1]

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, obs_mode, conv_lat_dim):
        super(Critic, self).__init__()
        self.obs_mode = obs_mode
        self.conv_lat_dim = conv_lat_dim
        if obs_mode != 'state':
            input_channels = 3 if self.obs_mode == 'rgb' else 4
            self.convolution = nn.Sequential(
                nn.Conv2d(input_channels, 16, 3, padding=1, bias=True), nn.ReLU(inplace=True),
                nn.MaxPool2d(4, 4), #if image_size[0] == 128 and image_size[1] == 128 else nn.MaxPool2d(2, 2),  # [32, 32]
                nn.Conv2d(16, 32, 3, padding=1, bias=True), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # [16, 16]
                nn.Conv2d(32, 64, 3, padding=1, bias=True), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # [8, 8]
                nn.Conv2d(64, 64, 3, padding=1, bias=True), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # [4, 4]
                nn.Conv2d(64, 64, 1, padding=0, bias=True), nn.ReLU(inplace=True),
                nn.Flatten(1),
                nn.Linear(1024, conv_lat_dim)
            )
        
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256) if obs_mode == 'state' else nn.Linear(state_dim + conv_lat_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l2_2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256) if obs_mode == 'state' else nn.Linear(state_dim + conv_lat_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l5_2 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action, img_state=None):
        
        if img_state is not None:
            n_e, bs, h, w, c = img_state.shape
            img_embeddings = self.convolution(img_state.reshape(n_e*bs, c, h, w)).reshape(n_e, bs, self.conv_lat_dim)
            state = torch.cat((state, img_embeddings), -1)
                
        
        sa = torch.cat([state, action], -1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l2_2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = F.relu(self.l5_2(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action, img_state=None):
        
        if img_state is not None:
            n_e, bs, h, w, c = img_state.shape
            img_embeddings = self.convolution(img_state.reshape(n_e*bs, c, h, w)).reshape(n_e, bs, self.conv_lat_dim)
            state = torch.cat((state, img_embeddings), -1)
                
        sa = torch.cat([state, action], -1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l2_2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
        self,
        num_envs,
        obs_mode,
        context_length,
        model_config,
        state_dim,
        action_dim,
        max_action,
        discount,
        tau,
        policy_noise,
        noise_clip,
        policy_freq,
        grad_clip,
        preload_weights=None
):

        if preload_weights == None:
            if model_config['critic_mode'] == 'FC':
                self.critic = Critic(state_dim, action_dim, obs_mode, model_config['conv_lat_dim']).to(device)
                self.critic_target = copy.deepcopy(self.critic).to(device)
                self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

            self.trans = Model(**model_config, state_dim=state_dim, act_dim=action_dim, obs_mode=obs_mode).to(device)
            self.trans_target = copy.deepcopy(self.trans).to(device)
            self.trans_optimizer = torch.optim.Adam(self.trans.parameters(), lr=3e-4)

        else:
            print('Found preload weights!')
            print('***DOWNLOADING WEIGHTS***')
            p2tr = preload_weights[0]
            self.trans = torch.load(p2tr).to(device)
            p2tr_tgt = preload_weights[1]
            self.trans_target = torch.load(p2tr_tgt).to(device)
            self.trans_optimizer = torch.optim.Adam(self.trans.parameters(), lr=3e-4)
            
            p2cr = preload_weights[2]
            self.critic = torch.load(p2cr).to(device)
            p2cr_tgt = preload_weights[3]
            self.critic_target = torch.load(p2cr_tgt).to(device)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        
        
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.context_length = context_length
        self.obs_mode = obs_mode
        self.total_it = 0
        self.eval_counter = 0
        self.trans_critic_mode = model_config['critic_mode']
        self.grad_clip = grad_clip
        

    def select_action(self, state): # state Tens(1, s_d)
        
        return self.actor(state).cpu().data.numpy()
    
    def stage_2_train(self, batch_size):
        
        self.total_it += 1
        train_batch = self.new_trans_RB.sample(batch_size)
        if self.obs_mode == 'state':
            states, actions, rewards, dones, next_states = train_batch
        else:
            states, actions, rewards, dones, next_states, img_states, img_next_states = train_batch 
            img_states, img_next_states = img_states.to(device).requires_grad_(True), img_next_states.to(device).requires_grad_(True)	
        
        states = states.to(device).requires_grad_(True)											#n_e, bs, context, state_dim
        actions = actions.to(device).requires_grad_(True)										#n_e, bs, action_dim
        rewards = rewards.to(device).requires_grad_(True)										#n_e, bs, 1
        dones = dones.to(device)											                    #n_e, bs, 1
        next_states = next_states.to(device).requires_grad_(True)								#n_e, bs, context, state_dim

        self.trans.train()
        if hasattr(self, 'critic'):
            self.critic.train()

        with torch.no_grad():
            noise = (                                                           #n_e, bs, a_d
                torch.randn_like(actions) * self.policy_noise  
            ).clamp(-self.noise_clip, self.noise_clip)
            
            if self.obs_mode == 'state':
                next_action = (
                    self.trans_target.actor_forward(next_states) + noise  			 #next_action = (n_e, bs, a_d)
                ).clamp(-self.max_action, self.max_action)
            else:
                next_action = self.trans_target.actor_forward(next_states, img_next_states)
                noise = ( torch.randn_like(next_action) * self.policy_noise ).clamp(-self.noise_clip, self.noise_clip)  
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
            
            if hasattr(self, 'critic'):
                target_Q1, target_Q2 = self.critic_target(next_states[:,:,-1,:], next_action) if self.obs_mode == 'state' else self.critic_target(next_states[:,:,-1,:], next_action, img_next_states[:,:,-1,])
            else:
                target_Q1, target_Q2 = self.trans_target.critic_forward(next_states, next_action)
            
            target_Q = torch.min(target_Q1, target_Q2)                                      #target_Q = (n_e, bs, 1)
            target_Q = rewards + (1-dones) * self.discount * target_Q       #target_Q = (n_e, bs, 1) + (n_e, bs, 1) * const * (n_e, bs, 1)
        
        if hasattr(self, 'critic'):
            current_Q1, current_Q2 = self.critic(states[:,:,-1,:], actions) if self.obs_mode == 'state' else self.critic(states[:,:,-1,:], actions, img_states[:,:,-1,])  #current_Q1 = (n_e, bs, 1)
        else:
            current_Q1, current_Q2 = self.trans.critic_forward(states, actions)


        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.experiment.add_scalar('Critic_loss', critic_loss.item(), self.total_it)

        if hasattr(self, 'critic'):
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            #torch.nn.utils.clip_grad_value_(self.critic.parameters(), self.grad_clip)
            critic_grad_norm = sum(p.grad.norm().item() for p in self.critic.parameters() if p.grad is not None)
        else:
            self.trans_critic_optimizer.zero_grad()
            critic_loss.backward()
            #torch.nn.utils.clip_grad_value_(self.critic_params, self.grad_clip)
            critic_grad_norm = sum(p.grad.norm().item() for p in self.tr_critic_params if p.grad is not None)

        
        self.experiment.add_scalar('critic_grad_norm', critic_grad_norm, self.total_it)
        
        if hasattr(self, 'critic'):
            self.critic_optimizer.step()
        else:
            self.trans_critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            if hasattr(self, 'critic'):
                trans_loss = -self.critic.Q1(states[:,:,-1,:], self.trans.actor_forward(states)).mean()
            else:
                trans_loss = -self.trans.Q1(states, self.trans.actor_forward(states)).mean()    
            
            self.experiment.add_scalar('Actor_loss', trans_loss, self.total_it)
            
            # Optimize the actor 
            self.trans_optimizer.zero_grad()
            trans_loss.backward()
            #torch.nn.utils.clip_grad_value_(self.actor_params, self.grad_clip)
            trans_grad_norm = sum(p.grad.norm().item() for p in self.trans.parameters() if p.grad is not None)
            self.experiment.add_scalar('actor_grad_norm', trans_grad_norm, self.total_it)

            self.trans_optimizer.step()
            
            
            # Update the frozen target models
            if hasattr(self, 'critic'):
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.trans.parameters(), self.trans_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    
############################################################################################################################

class New_Trans_RB():
    def __init__(self, num_envs, size, context, state_dim, act_dim, obs_mode):
        self.size = size
        self.context = context
        self.idx = 0
        self.overfilled = False
        self.obs_mode = obs_mode

        self.observations = torch.zeros((num_envs, size, context, state_dim), dtype=torch.float32)
        self.actions = torch.zeros((num_envs, size, act_dim), dtype=torch.float32)
        self.returns = torch.zeros((num_envs, size, 1), dtype=torch.float32)
        self.dones = torch.zeros((num_envs, size, 1), dtype=torch.float32)
        self.next_observations = torch.zeros((num_envs, size, context, state_dim), dtype=torch.float32)
        
        if obs_mode != 'state':
            channels = 3 if obs_mode == 'rgb' else 4
            self.img_observations = torch.zeros((num_envs, size, context, 128, 128, channels), dtype=torch.float32)
            self.img_next_observations = torch.zeros((num_envs, size, context, 128, 128, channels), dtype=torch.float32)
            
    
    def recieve_traj(self, obs, acts, rets, dones, next_obs, img_obs=None, img_next_obs=None):
        '''
        :
        obs = [....,Tens(n_e, s_d),....]
        acts = Tens(n_e, a_d)
        rets = Tens(n_e, 1)
        dones = Tens(n_e, 1)
        next_obs = [....,Tens(n_e, s_d),....]
        '''

        obs = torch.stack(obs, dim=1)                           # n_e, cont, s_d
        acts = acts.to(torch.float32)                           # n_e, a_d
        rets = rets.to(torch.float32)                           # n_e, 1
        dones = dones                                           # n_e, 1
        next_obs = torch.stack(next_obs, dim=1)                 # n_e, cont, s_d
        if self.obs_mode != 'state':
            img_obs = torch.stack(img_obs, dim=1).float() 
            img_next_obs = torch.stack(img_next_obs, dim=1).float() 
            img_obs[:, :, :, :, :3] /= 225.0
            img_next_obs[:, :, :, :, :3] /= 225.0
            self.img_observations[:,self.idx,] = img_obs
            self.img_next_observations[:,self.idx,] = img_next_obs
        self.observations[:,self.idx,] = obs
        self.actions[:,self.idx,] = acts
        self.returns[:,self.idx,] = rets
        self.dones[:,self.idx,] = dones
        self.next_observations[:,self.idx,] = next_obs

        self.idx += 1
        if self.idx >= self.size:
            self.idx = 0
            self.overfilled = True

    def sample(self, batch_size):
        if batch_size > self.size:
            raise ValueError("batch > size")
        
        elif (batch_size >= self.idx) and (self.overfilled == False):
            idxs = torch.randperm(self.idx) #
            
        elif (batch_size >= self.idx) and (self.overfilled == True):
            idxs = torch.randperm(self.size)[:batch_size]
        
        elif (batch_size < self.idx) and (self.overfilled == False):
            idxs = torch.randperm(self.idx)[:batch_size]
        
        elif (batch_size < self.idx) and (self.overfilled == True):
            idxs = torch.randperm(self.size)[:batch_size]
        
        batch = (
            self.observations[:,idxs, ],
            self.actions[:,idxs, ],
            self.returns[:,idxs, ],
            self.dones[:,idxs, ],
            self.next_observations[:,idxs, ],
        ) if self.obs_mode == 'state' else (
            self.observations[:,idxs, ],
            self.actions[:,idxs, ],
            self.returns[:,idxs, ],
            self.dones[:,idxs, ],
            self.next_observations[:,idxs, ],
            self.img_observations[:,idxs, ],
            self.img_next_observations[:,idxs, ]
        ) 

        return batch
