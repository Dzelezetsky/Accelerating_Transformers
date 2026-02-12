import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gymnasium import spaces
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import gym


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
        #return {'state': obs_tensor}
        return obs_tensor
        
    def step(self, action):
        
        obs, reward, done, info = self.env.step(action)
        
        
        obs_tensor = self._to_tensor(obs).unsqueeze(0)
        reward_tensor = self._to_tensor(np.array(reward, dtype=np.float32))
        done_tensor = self._to_tensor(np.array(done, dtype=np.bool_))
        
        
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



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l2_2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l2_2(a))
        return self.max_action * torch.tanh(self.l3(a))


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
    

############################################################################################################################
# Transformer actor+critic for acceleration (TD3-style)
############################################################################################################################

class SinusoidalPositionalEncoding(nn.Module):
    """
    Adds fixed sinusoidal positional encoding to a (B, T, D) tensor.
    Cached per (device, dtype, T, D).
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = int(d_model)
        self.register_buffer("_pe", torch.empty(0), persistent=False)

    def _build(self, T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        position = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)  # (T, 1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=dtype)
            * (-np.log(10000.0) / self.d_model)
        )  # (D/2,)
        pe = torch.zeros(T, self.d_model, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = int(x.size(1))
        if (
            self._pe.numel() == 0
            or self._pe.size(1) < T
            or self._pe.size(2) != self.d_model
            or self._pe.device != x.device
            or self._pe.dtype != x.dtype
        ):
            self._pe = self._build(T, x.device, x.dtype)
        return x + self._pe[:, :T, :]


class TransformerModel(nn.Module):
    """
    Minimal Transformer encoder model with:
      - actor_forward(states, img_states=None) -> actions
      - critic_forward(states, actions, img_states=None) -> (Q1, Q2)
      - Q1(states, actions, img_states=None) -> Q1

    Shapes expected by your code:
      states: (n_e, bs, context, state_dim)
      actions: (n_e, bs, act_dim)
      img_states (optional): (n_e, bs, context, H, W, C) with NHWC layout
    """
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        obs_mode: str = "state",
        max_action: float = 1.0,
        # Common config keys (extra keys are ignored via **kwargs):
        d_model = None,
        nhead = None,
        num_layers = None,
        dropout = None,
        dim_feedforward = None,
        conv_lat_dim = 64,
        critic_hidden = 256,
        critic_mode = "shared",
        **kwargs,
    ):
        super().__init__()

        # Backward/legacy config aliases
        d_model = d_model or kwargs.get("embed_dim") or kwargs.get("n_embd") or kwargs.get("hidden_dim") or 128
        nhead = nhead or kwargs.get("n_head") or kwargs.get("n_heads") or 4
        num_layers = num_layers or kwargs.get("n_layer") or kwargs.get("n_layers") or kwargs.get("depth") or 4
        dropout = float(dropout if dropout is not None else kwargs.get("dropout", 0.1))
        dim_feedforward = dim_feedforward or kwargs.get("ff_dim") or (4 * int(d_model))

        self.state_dim = int(state_dim)
        self.act_dim = int(act_dim)
        self.obs_mode = str(obs_mode)
        self.max_action = float(max_action)
        self.critic_mode = str(critic_mode)

        self.use_img = self.obs_mode != "state"
        self.conv_lat_dim = int(conv_lat_dim)

        if self.use_img:
            input_channels = 3 if self.obs_mode == "rgb" else 4
            self.img_encoder = nn.Sequential(
                nn.Conv2d(input_channels, 16, 3, padding=1, bias=True), nn.ReLU(inplace=True),
                nn.MaxPool2d(4, 4),
                nn.Conv2d(16, 32, 3, padding=1, bias=True), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, 3, padding=1, bias=True), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 64, 3, padding=1, bias=True), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 64, 1, padding=0, bias=True), nn.ReLU(inplace=True),
                nn.Flatten(1),
                nn.Linear(1024, self.conv_lat_dim),
            )
            in_dim = self.state_dim + self.conv_lat_dim
        else:
            self.img_encoder = None
            in_dim = self.state_dim

        self.in_proj = nn.Linear(in_dim, int(d_model))
        self.pos_enc = SinusoidalPositionalEncoding(int(d_model))
        self.ln_in = nn.LayerNorm(int(d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))

        self._mask_cache_T = -1
        self._mask_cache = None

        self.actor_head = nn.Sequential(
            nn.Linear(int(d_model), int(d_model)),
            nn.ReLU(inplace=True),
            nn.Linear(int(d_model), self.act_dim),
        )

        q_in = int(d_model) + self.act_dim
        self.q1 = nn.Sequential(
            nn.Linear(q_in, critic_hidden), nn.ReLU(inplace=True),
            nn.Linear(critic_hidden, critic_hidden), nn.ReLU(inplace=True),
            nn.Linear(critic_hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(q_in, critic_hidden), nn.ReLU(inplace=True),
            nn.Linear(critic_hidden, critic_hidden), nn.ReLU(inplace=True),
            nn.Linear(critic_hidden, 1),
        )

    def _causal_mask(self, T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # nn.Transformer expects float mask with -inf in masked positions
        if self._mask_cache is None or self._mask_cache_T != T or self._mask_cache.device != device or self._mask_cache.dtype != dtype:
            self._mask_cache_T = int(T)
            self._mask_cache = torch.triu(
                torch.full((T, T), float("-inf"), device=device, dtype=dtype),
                diagonal=1,
            )
        return self._mask_cache

    def _flatten_states(self, x: torch.Tensor):
        if x.dim() == 4:
            n_e, bs, T, D = x.shape
            return x.reshape(n_e * bs, T, D), int(n_e), int(bs)
        if x.dim() == 3:
            B, T, D = x.shape
            return x, 1, int(B)
        raise ValueError(f"Expected states dim 3 or 4, got shape={tuple(x.shape)}")

    def _encode(self, states: torch.Tensor, img_states: torch.Tensor = None) -> torch.Tensor:
        x, n_e, bs = self._flatten_states(states)  # (B, T, state_dim)
        B, T, _ = x.shape

        if self.use_img:
            if img_states is None:
                img_feat = torch.zeros((B, T, self.conv_lat_dim), device=x.device, dtype=x.dtype)
            else:
                if img_states.dim() != 6:
                    raise ValueError(f"Expected img_states dim 6 (n_e, bs, T, H, W, C), got shape={tuple(img_states.shape)}")
                n_e2, bs2, T2, H, W, C = img_states.shape
                img_bt = img_states.reshape(n_e2 * bs2 * T2, H, W, C).permute(0, 3, 1, 2).contiguous()
                img_emb = self.img_encoder(img_bt)  # (B*T, conv_lat_dim)
                img_feat = img_emb.reshape(n_e2 * bs2, T2, self.conv_lat_dim)

            x = torch.cat([x, img_feat], dim=-1)

        x = self.in_proj(x)
        x = self.pos_enc(x)
        x = self.ln_in(x)

        mask = self._causal_mask(T, x.device, x.dtype)
        h = self.encoder(x, mask=mask)  # (B, T, d_model)
        return h.reshape(n_e, bs, T, h.size(-1))

    def actor_forward(self, states: torch.Tensor, img_states = None) -> torch.Tensor:
        h = self._encode(states, img_states)  # (n_e, bs, T, d_model)
        last = h[:, :, -1, :]                # (n_e, bs, d_model)
        a = self.actor_head(last)
        return self.max_action * torch.tanh(a)

    def critic_forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        img_states = None,
    ):
        h = self._encode(states, img_states)  # (n_e, bs, T, d_model)
        last = h[:, :, -1, :]                 # (n_e, bs, d_model)
        sa = torch.cat([last, actions], dim=-1)
        return self.q1(sa), self.q2(sa)

    def Q1(self, states: torch.Tensor, actions: torch.Tensor, img_states = None) -> torch.Tensor:
        h = self._encode(states, img_states)
        last = h[:, :, -1, :]
        sa = torch.cat([last, actions], dim=-1)
        return self.q1(sa)



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
        grad_clip
):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        

        self.critic = Critic(state_dim, action_dim, obs_mode, model_config['conv_lat_dim']).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        

        self.trans = TransformerModel(
            state_dim=state_dim,
            act_dim=action_dim,
            obs_mode=obs_mode,
            max_action=max_action,
            **model_config,
        ).to(device)

        self.trans_target = copy.deepcopy(self.trans).to(device)
        self.trans_optimizer = torch.optim.Adam(self.trans.parameters(), lr=3e-4)
        self.trans_RB = Trans_RB(num_envs, 30000, context_length, state_dim, action_dim) 


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
        #state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        
        return self.actor(state).cpu().data.numpy()
    
    def stage_2_train(self, batch_size):
        '''
        Function for transformer training on the second stage
        '''
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
            
            if hasattr(self, 'critic_target'):
                
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
            torch.nn.utils.clip_grad_value_(self.critic.parameters(), self.grad_clip)
            critic_grad_norm = sum(p.grad.norm().item() for p in self.critic.parameters() if p.grad is not None)
        else:
            self.trans_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_value_(self.trans.parameters(), self.grad_clip)
            critic_grad_norm = sum(p.grad.norm().item() for p in self.trans.parameters() if p.grad is not None)

        
        self.experiment.add_scalar('critic_grad_norm', critic_grad_norm, self.total_it)
        
        if hasattr(self, 'critic'):
            self.critic_optimizer.step()
        else:
            self.trans_optimizer.step()

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
            torch.nn.utils.clip_grad_value_(self.trans.parameters(), self.grad_clip)
            trans_grad_norm = sum(p.grad.norm().item() for p in self.trans.parameters() if p.grad is not None)
            self.experiment.add_scalar('actor_grad_norm', trans_grad_norm, self.total_it)

            self.trans_optimizer.step()
            
                    

            # Update the frozen target models
            if hasattr(self, 'critic'):
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.trans.parameters(), self.trans_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)






    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        

        with torch.no_grad():
            
            noise = (
                torch.randn_like(action) * self.policy_noise        #noise = (n_e, b_s, a_d)
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target(next_state) + noise               #next_action = (n_e, b_s, a_d)
            ).clamp(-self.max_action, self.max_action)
            
            
            

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)                          #target_Q = (n_e, b_s, 1)
            target_Q = reward + not_done * self.discount * target_Q             #target_Q = (n_e, b_s, 1) + (n_e, b_s, 1) * const * (n_e, b_s, 1)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)                     #current_Q1(and Q2) = (n_e, b_s, 1)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()    #state=(n_e,b_s,s_d), actor(state)=(n_e,b_s,a_d), actor_loss=(n_e,b_s,1)
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        
        
    def train_trans_actor(self, batch_size, additional_ascent):
        self.trans.train()
        idxs = torch.randperm(self.trans_RB.idx)
        chunks = split_indices(idxs, batch_size)
        bc_losses = []
        ascent_losses = []
        for chunk in chunks:
            # BEHAVIOR CLONNING
            batch = self.trans_RB.sample(chunk)
            states = batch[0]
            targets = batch[2]
            preds = self.trans.actor_forward(states)
            loss = nn.MSELoss()(preds, targets)
            self.trans_optimizer.zero_grad()
            bc_losses.append(loss.item())
            loss.backward()
            self.trans_optimizer.step()
            
            # GRADIENT ASCENT
            if additional_ascent:
                trans_loss = -self.critic.Q1(states[:,:,-1,:], self.trans.actor_forward(states)).mean()
                ascent_losses.append(trans_loss.cpu().detach().numpy())
                self.trans_optimizer.zero_grad()
                trans_loss.backward()
                self.trans_optimizer.step()
                
        self.trans_RB.reset()
        self.experiment.add_scalar('Trans_BC_loss', np.mean(bc_losses), self.eval_counter)
        if additional_ascent:
            self.experiment.add_scalar('Trans_Online_loss', np.mean(ascent_losses), self.eval_counter)
        self.eval_counter += 1	
            
        
        for param, target_param in zip(self.trans.parameters(), self.trans_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()    
    
    def train_trans_critic(self, batch_size, additional_bellman):
        self.trans.train()
        idxs = torch.randperm(self.trans_RB.idx)
        chunks = split_indices(idxs, batch_size)
        bc_losses = []
        bellman_losses = []
        for chunk in chunks:
            # BEHAVIOR CLONNING
            batch = self.trans_RB.sample(chunk)
            states = batch[0]       # n_e, b_s, cont, s_d
            next_states = batch[1]
            actions = batch[2]      # n_e, b_s, a_d
            rewards = batch[3]      # n_e, b_s, 1
            not_dones = batch[4]    # n_e, b_s, 1
            target1, target2 = self.critic(states[:,:,-1,:], actions)
            pred1, pred2 = self.trans.critic_forward(states, actions)  # pred1=pred2 = n_e, b_s, 1
            loss = nn.MSELoss()(pred1, target1) + nn.MSELoss()(pred2, target2)
            self.trans_optimizer.zero_grad()
            bc_losses.append(loss.item())
            loss.backward()
            self.trans_optimizer.step()
            
            if additional_bellman:
                with torch.no_grad():
                    noise = (
                        torch.randn_like(actions) * self.policy_noise        #noise = (n_e, b_s, a_d)
                    ).clamp(-self.noise_clip, self.noise_clip)
                    next_actions = (
                        self.trans_target.actor_forward(next_states) + noise               #next_action = (n_e, b_s, a_d)
                    ).clamp(-self.max_action, self.max_action)
                    
                    # Compute the target Q value
                    target_Q1, target_Q2 = self.trans_target.critic_forward(next_states, next_actions)
                    target_Q = torch.min(target_Q1, target_Q2)                          #target_Q = (n_e, b_s, 1)
                    target_Q = rewards + not_dones * self.discount * target_Q             #target_Q = (n_e, b_s, 1) + (n_e, b_s, 1) * const * (n_e, b_s, 1)
            
                # Get current Q estimates
                current_Q1, current_Q2 = self.trans.critic_forward(states, actions)                     #current_Q1(and Q2) = (n_e, b_s, 1)

                # Compute critic loss
                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) #

                # Optimize the critic
                self.trans_optimizer.zero_grad()
                bellman_losses.append(loss.item())
                critic_loss.backward()
                self.trans_optimizer.step()
                
        
        self.experiment.add_scalar('Trans_Critic_BC_loss', np.mean(bc_losses), self.eval_counter) 
        if additional_bellman:
            self.experiment.add_scalar('Trans_Critic_Bellman_loss', np.mean(bellman_losses), self.eval_counter) 
    
    
############################################################################################################################
class ReplayBuffer(object):
    
    def __init__(self, num_envs, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((num_envs, max_size, state_dim))
        self.action = np.zeros((num_envs, max_size, action_dim))
        self.next_state = np.zeros((num_envs, max_size, state_dim))
        self.reward = np.zeros((num_envs, max_size, 1))
        self.not_done = np.zeros((num_envs, max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        


    def add(self, state, action, next_state, reward, done):
        self.state[:,self.ptr,] = state
        self.action[:,self.ptr,] = action
        self.next_state[:,self.ptr,] = next_state
        self.reward[:,self.ptr,] = reward.reshape(-1,1)
        self.not_done[:,self.ptr,] = 1. - done.reshape(-1,1)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[:,ind,]).to(self.device),
            torch.FloatTensor(self.action[:,ind,]).to(self.device),
            torch.FloatTensor(self.next_state[:,ind,]).to(self.device),
            torch.FloatTensor(self.reward[:,ind,]).to(self.device),
            torch.FloatTensor(self.not_done[:,ind,]).to(self.device)
        )


class Trans_RB(object):
    def __init__(self, num_envs, size, context, state_dim, act_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.size = size
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.context = context
        self.idx = 0
        self.overfilled = False
        self.num_envs = num_envs

        
        self.observations = torch.zeros((num_envs, size, context, state_dim), dtype=torch.float32).to(self.device)  # n_e, size, cont, s_d
        self.next_observations = torch.zeros((num_envs, size, context, state_dim), dtype=torch.float32).to(self.device)
        self.actions = torch.zeros((num_envs, size, act_dim), dtype=torch.float32).to(self.device)                   # n_e, size, a_d
        self.rewards = torch.zeros((num_envs, size, 1), dtype=torch.float32).to(self.device)    
        self.not_dones = torch.zeros((num_envs, size, 1), dtype=torch.float32).to(self.device)
    
    def recieve_traj(self, obs, next_obs, acts, rews, n_dones):
        ''' 
        obs list(...,tensor(n_e, ),....)
        '''

        obs = torch.stack(obs, dim=1)    # n_e, cont, s_d
        next_obs = torch.stack(next_obs, dim=1)    # n_e, cont, s_d
        acts = acts.to(torch.float32)    # n_e, a_d
        rews = rews.to(torch.float32) 
        n_dones = n_dones.to(torch.float32)
        
        self.observations[:,self.idx,] = obs.to(device)
        self.next_observations[:,self.idx,] = next_obs.to(device)
        self.actions[:,self.idx] = acts.to(device)
        self.rewards[:,self.idx,] = rews.to(device)
        self.not_dones[:,self.idx,] = n_dones.to(device)

        self.idx += 1
        if self.idx >= self.size:
            self.idx = 0
            self.overfilled = True

    def sample(self, idxs):
        
        batch = (
            self.observations[:,idxs, ],  # n_e, b_s, cont, s_d
            self.next_observations[:,idxs, ],
            self.actions[:,idxs, ],        # n_e, b_s, a_d
            self.rewards[:,idxs, ],         # n_e, b_s, 1
            self.not_dones[:,idxs, ]        # n_e, b_s, 1
        )

        return batch

    def reset(self):
        
        
        self.idx = 0
        self.overfilled = False
        self.observations = torch.zeros((self.num_envs, self.size, self.context, self.state_dim), dtype=torch.float32).to(self.device)  # n_e, size, cont, s_d
        self.actions = torch.zeros((self.num_envs, self.size, self.act_dim), dtype=torch.float32).to(self.device)
        self.rewards = torch.zeros((self.num_envs, self.size, 1), dtype=torch.float32).to(self.device)    
        self.not_dones = torch.zeros((self.num_envs, self.size, 1), dtype=torch.float32).to(self.device)


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
            idxs = torch.randperm(self.idx) 
            
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
        
        

def split_indices(indices, chunk_size):
    # Если размер индексов меньше, чем размер порции, просто возвращаем их как один блок
    if len(indices) <= chunk_size:
        return [indices]
    
    # Разделяем индексы на порции фиксированного размера
    chunks = torch.split(indices, chunk_size)

    return chunks        