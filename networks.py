import torch
import torch.nn as nn
import torch.nn.functional as F

class ObservationEncoder(nn.Module):
    """
    Observation Encoder for 2D AUV scenario - processes SELF + TEAMMATE + ENV observations.
    
    Your environment constructs observations in this order:
    1. Own state (3):        [x, y, ψ]              - indices [0:3]
    2. Nearby defenders (2): [rel_x, rel_y]         - indices [3:5] - TEAMMATE BRANCH
    3. Known attackers (2):  [rel_x, rel_y]         - indices [5:7] - BOTH SELF & TEAMMATE  
    4. Target info (3):      [Δx_tgt, Δy_tgt, R_tgt] - indices [7:10]
    5. Boundary info (3):    [Δx_bnd, Δy_bnd, R_bnd] - indices [10:13]
    6. Local obstacles (40): [obstacle vertices...]  - indices [13:]
    
    This encoder processes three observation types:
      - self-obs: own state + target + boundary + attackers (11 dims) → f_self
      - teammate-obs: nearby defenders + known attackers (4 dims) → f_team  
      - env-obs: local obstacles (40 dims) → f_env
    
    Final output: õ_i = [f_self(o_i^self) || f_team(o_i^team) || f_env(o_i^env)]
    
    Note: Attackers appear in BOTH branches for different purposes:
    - In SELF: direct threat assessment ("how do attackers threaten ME?")  
    - In TEAMMATE: spatial situational awareness ("how far are attackers from my teammates?")
    
    The teammate observations get used TWICE:
    1. Here: to shape each agent's own embedding
    2. Copy-and-Concat: full embeddings used for cross-agent awareness [õ_i || õ_j]
    """
    def __init__(self,
                 obs_dim_per_defender: int,
                 max_neighbors: int = 1,
                 max_attackers: int = 1,
                 d_self: int = 64,
                 d_team: int = 32, 
                 d_env: int = 64,
                 env_channels: int = 16,
                 conv_kernel: int = 3):
        super().__init__()
        
        # Define observation structure based on your environment
        self.own_state_dim = 3
        self.nearby_defenders_dim = 2 * max_neighbors  # 2
        self.known_attackers_dim = 2 * max_attackers   # 2  
        self.target_info_dim = 3
        self.boundary_info_dim = 3
        self.obstacles_dim = obs_dim_per_defender - (
            self.own_state_dim + self.nearby_defenders_dim + 
            self.known_attackers_dim + self.target_info_dim + self.boundary_info_dim
        )  # 40 for your scenario
        
        # Calculate slice indices for all observation components
        self.own_state_slice = slice(0, 3)  # [0:3]
        self.nearby_defenders_slice = slice(3, 3 + self.nearby_defenders_dim)  # [3:5]
        self.known_attackers_slice = slice(5, 5 + self.known_attackers_dim)   # [5:7] 
        self.target_info_slice = slice(7, 10)   # [7:10]
        self.boundary_info_slice = slice(10, 13)  # [10:13]
        self.obstacles_slice = slice(13, None)    # [13:]
        
        # Dimensions for the three processing branches (REORGANIZED)
        self.self_dim = self.own_state_dim + self.target_info_dim + self.boundary_info_dim + self.known_attackers_dim  # 11
        self.teammate_dim = self.nearby_defenders_dim + self.known_attackers_dim  # 2 + 2 = 4
        self.env_dim = self.obstacles_dim  # 40
        
        # Three separate processing branches
        
        # Self-MLP: f_self(o_i^self) - includes threats (attackers)
        self.self_fc = nn.Sequential(
            nn.Linear(self.self_dim, d_self),
            nn.LeakyReLU(0.01)
        )
        
        # Teammate-MLP: f_team(o_i^team) - spatial awareness of teammates & attacker positions
        self.teammate_fc = nn.Sequential(
            nn.Linear(self.teammate_dim, d_team),
            nn.LeakyReLU(0.01)
        )
        
        # Env Conv + MLP: f_env(o_i^env)
        self.env_conv = nn.Conv1d(
            in_channels=1,
            out_channels=env_channels,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2
        )
        self.env_fc = nn.Sequential(
            nn.Linear(env_channels * self.env_dim, d_env),
            nn.LeakyReLU(0.01)
        )
        
        # Store output dimensions
        self.output_dim = d_self + d_team + d_env  # 64 + 32 + 64 = 160
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
          obs: Tensor of shape (batch_size, obs_dim_per_defender)
        Returns:
          obs_emb: Tensor of shape (batch_size, d_self + d_team + d_env)
                   õ_i = [f_self || f_team || f_env]
        """
        # Extract all observation components
        own_state = obs[:, self.own_state_slice]        # (batch, 3)
        nearby_defenders = obs[:, self.nearby_defenders_slice]  # (batch, 2) - teammates relative to self
        known_attackers = obs[:, self.known_attackers_slice]    # (batch, 2) - attackers relative to self
        target_info = obs[:, self.target_info_slice]    # (batch, 3)
        boundary_info = obs[:, self.boundary_info_slice] # (batch, 3)
        obstacles = obs[:, self.obstacles_slice]        # (batch, 40)
        
        # Compute attacker positions relative to teammate
        # attacker_rel_to_teammate = attacker_rel_to_self - teammate_rel_to_self
        attackers_rel_to_teammate = known_attackers - nearby_defenders  # (batch, 2)
        
        # Concatenate observations for each branch (REORGANIZED)
        self_obs = torch.cat([own_state, target_info, boundary_info, known_attackers], dim=-1)  # (batch, 11) - attackers relative to SELF
        teammate_obs = torch.cat([nearby_defenders, attackers_rel_to_teammate], dim=-1)  # (batch, 4) - teammates relative to SELF, attackers relative to TEAMMATE
        env_obs = obstacles  # (batch, 40)
        
        # Process each branch: f_self, f_team, f_env
        self_emb = self.self_fc(self_obs)     # (batch, d_self)
        teammate_emb = self.teammate_fc(teammate_obs)  # (batch, d_team)
        
        # Env branch: Conv1d + MLP
        env_in = env_obs.unsqueeze(1)     # (batch, 1, env_dim)
        conv_out = self.env_conv(env_in)  # (batch, env_ch, env_dim)
        conv_out = F.leaky_relu(conv_out, 0.01)
        conv_flat = conv_out.view(conv_out.size(0), -1)  # (batch, env_ch*env_dim)
        env_emb = self.env_fc(conv_flat)  # (batch, d_env)
        
        # Final concatenation: õ_i = [f_self || f_team || f_env]
        obs_emb = torch.cat([self_emb, teammate_emb, env_emb], dim=-1)  
        # (batch, d_self + d_team + d_env) = (batch, 160)
        
        return obs_emb


class CopyConcat(nn.Module):
    """
    Copy-and-Concat module as in Fig.9/Eq.(17) from the paper:
    
    Input:  x of shape (batch_size, N, D) - all agents' embeddings
    Output: out of shape (batch_size, N, D) - processed embeddings
    
    For each agent i:
        - Gather all x[:, j, :] for j != i → concatenate into (N-1)*D vector
        - Project via Linear((N-1)*D → D) + ReLU
        
    This matches the paper's approach exactly: collect teammates' embeddings,
    project them back to original dimension with learned transformation.
    """
    def __init__(self, n_agents: int, emb_dim: int):
        """
        Args:
            n_agents: Number of agents (N) - 2 for your scenario
            emb_dim: Embedding dimension (D) - 160 from 3-branch ObservationEncoder
        """
        super().__init__()
        self.n = n_agents
        self.D = emb_dim
        # (N-1)*D -> D projection with learnable parameters
        self.fc = nn.Linear((n_agents - 1) * emb_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) - batch of agent embeddings
        Returns:
            out: (B, N, D) - processed embeddings ready for attention
        """
        B, N, D = x.shape
        assert N == self.n and D == self.D, f"Expected ({B}, {self.n}, {self.D}), got {x.shape}"

        out = torch.zeros_like(x)  # (B, N, D)
        
        for i in range(N):
            # Collect all teammates j != i
            others = [x[:, j, :] for j in range(N) if j != i]   # List of (B, D) tensors
            cat = torch.cat(others, dim=-1)                     # (B, (N-1)*D)
            proj = F.relu(self.fc(cat))                         # (B, D) with learned projection
            out[:, i, :] = proj
            
        return out  # (B, N, D)


class RSABlock(nn.Module):
    """
    Residual Self-Attention (RSA) Block as described in the paper.
    
    This is the core communication mechanism that allows each agent to fuse
    information from teammates in a permutation-invariant, learned manner.
    
    Architecture (following Fig. 9 in paper):
    1. LayerNorm on per-agent embeddings
    2. Multi-Head Self-Attention (Q, K, V projections)
    3. First Residual Add (attention + input)
    4. Second LayerNorm + Feed-Forward Network (D→4D→D)
    5. Second Residual Add (FFN + previous)
    6. Global Context Pooling (average across agents)
    
    Input:  (B, N, input_dim) - batch of agent embeddings (can be 2D for concatenated inputs)
    Output: 
        - refined: (B, N, input_dim) - per-agent embeddings with teammate information
        - z: (B, input_dim) - global context vector for critic
    """
    
    def __init__(self, input_dim: int = 320, num_heads: int = 4, dropout: float = 0.1):
        """
        Args:
            input_dim: Input embedding dimension (2D=320 for concatenated original+copyconcat)
            num_heads: Number of attention heads - 4 as suggested
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        
        # Ensure input_dim is divisible by num_heads for multi-head attention
        assert input_dim % num_heads == 0, f"input_dim {input_dim} must be divisible by num_heads {num_heads}"
        
        # 1. First LayerNorm (before attention)
        self.norm1 = nn.LayerNorm(input_dim)
        
        # 2. Multi-Head Self-Attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Input shape: (B, N, input_dim)
        )
        
        # 3. Second LayerNorm (before FFN)
        self.norm2 = nn.LayerNorm(input_dim)
        
        # 4. Feed-Forward Network (input_dim → 4*input_dim → input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * input_dim, input_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, input_dim) - agent embeddings (original + copyconcat concatenated)
        
        Returns:
            refined: (B, N, input_dim) - per-agent embeddings with teammate info
            z: (B, input_dim) - global context vector for critic
        """
        B, N, D = x.shape
        assert D == self.input_dim, f"Expected input_dim={self.input_dim}, got {D}"
        
        # 1. First LayerNorm
        normed1 = self.norm1(x)  # (B, N, input_dim)
        
        # 2. Multi-Head Self-Attention
        # Each agent attends to all agents (including itself)
        attn_output, attn_weights = self.multihead_attn(
            query=normed1,    # (B, N, input_dim)
            key=normed1,      # (B, N, input_dim)  
            value=normed1,    # (B, N, input_dim)
            need_weights=False  # We don't need attention weights for now
        )
        
        # 3. First Residual Connection
        residual1 = x + attn_output  # (B, N, input_dim)
        
        # 4. Second LayerNorm
        normed2 = self.norm2(residual1)  # (B, N, input_dim)
        
        # 5. Feed-Forward Network
        ffn_output = self.ffn(normed2)  # (B, N, input_dim)
        
        # 6. Second Residual Connection
        refined = residual1 + ffn_output  # (B, N, input_dim)
        
        # 7. Global Context Pooling
        # Average across all N agents to get global context
        z = refined.mean(dim=1)  # (B, input_dim)
        
        return refined, z


class ActorHead(nn.Module):
    """
    Actor Head g_act from the paper.
    
    Takes each agent's refined embedding o_hat_i ∈ R^(2D) (from RSA) together with 
    the global context z ∈ R^(2D) and produces a Gaussian policy over continuous 
    actions (v, ω) for AUV control.
    
    Mathematical specification:
    1. Joint input: h_i,t^(0) = [o_hat_i,t || z_t] ∈ R^(4D) (since both are 2D now)
    2. MLP: h_i,t^(0) → h_i,t^(1) → ... → h_i,t^(L) ∈ R^H'
    3. Policy heads: 
       - μ_i,t = W_μ h_i,t + b_μ ∈ R^2 (means)
       - log σ_i,t = W_σ h_i,t + b_σ ∈ R^2 (log-stds)
    4. Gaussian policy: a_i,t ∼ N(μ_i,t, diag(σ_i,t^2))
    5. Optional tanh-squashing for actuator limits
    """
    
    def __init__(self, 
                 rsa_output_dim: int = 320,  # 2D from concatenated RSA output
                 hidden_sizes: list = [128, 64],
                 action_dim: int = 2,
                 use_tanh: bool = True,
                 v_max: float = 5.0,
                 omega_max: float = 0.5):
        """
        Args:
            rsa_output_dim: RSA output dimension (2D=320 for concatenated original+copyconcat)
            hidden_sizes: List of hidden layer sizes [H, H'] 
            action_dim: Action dimension (2 for v, ω)
            use_tanh: Whether to use tanh-squashing for actuator limits
            v_max: Maximum velocity for defender
            omega_max: Maximum angular velocity for defender
        """
        super().__init__()
        self.rsa_output_dim = rsa_output_dim
        self.action_dim = action_dim
        self.use_tanh = use_tanh
        self.v_max = v_max
        self.omega_max = omega_max
        
        # Input dimension: [refined_embedding || global_context] = 2D + 2D = 4D
        input_dim = 2 * rsa_output_dim
        
        # Multi-layer perceptron (MLP)
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.final_hidden_dim = prev_dim
        
        # Policy heads
        self.mean_head = nn.Linear(self.final_hidden_dim, action_dim)
        self.log_std_head = nn.Linear(self.final_hidden_dim, action_dim)
        
        # Initialize policy heads
        self._init_policy_heads()
    
    def _init_policy_heads(self):
        """Initialize policy heads with small weights for stable training"""
        # Small initialization for mean head
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0.0)
        
        # Initialize log_std to reasonable values
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
        nn.init.constant_(self.log_std_head.bias, -0.5)  # Initial std ≈ 0.6
    
    def forward(self, refined_embeddings: torch.Tensor, global_context: torch.Tensor):
        """
        Args:
            refined_embeddings: (B, N, 2D) - per-agent embeddings from RSA
            global_context: (B, 2D) - global context from RSA
        
        Returns:
            policy_params: dict containing:
                - 'means': (B, N, action_dim) - μ_i,t for each agent
                - 'log_stds': (B, N, action_dim) - log σ_i,t for each agent  
                - 'stds': (B, N, action_dim) - σ_i,t for each agent
        """
        B, N, rsa_dim = refined_embeddings.shape
        assert rsa_dim == self.rsa_output_dim, f"Expected rsa_output_dim={self.rsa_output_dim}, got {rsa_dim}"
        
        # Expand global context to match number of agents
        # (B, 2D) → (B, N, 2D)
        global_expanded = global_context.unsqueeze(1).expand(B, N, self.rsa_output_dim)
        
        # 1. Joint input formation: [o_hat_i || z] for each agent
        joint_input = torch.cat([refined_embeddings, global_expanded], dim=-1)  # (B, N, 4D)
        
        # 2. MLP processing for each agent
        # Flatten to process all agents in parallel: (B*N, 4D)
        joint_flat = joint_input.view(B * N, -1)
        hidden = self.mlp(joint_flat)  # (B*N, H')
        
        # 3. Policy parameter heads
        means_flat = self.mean_head(hidden)  # (B*N, action_dim)
        log_stds_flat = self.log_std_head(hidden)  # (B*N, action_dim)
        
        # Reshape back to per-agent format
        means = means_flat.view(B, N, self.action_dim)  # (B, N, action_dim)
        log_stds = log_stds_flat.view(B, N, self.action_dim)  # (B, N, action_dim)
        
        # Clamp log_stds for numerical stability
        log_stds = torch.clamp(log_stds, min=-20, max=2)
        stds = torch.exp(log_stds)  # σ_i,t = exp(log σ_i,t)
        
        return {
            'means': means,
            'log_stds': log_stds,
            'stds': stds
        }
    
    def sample_actions(self, policy_params: dict, deterministic: bool = False):
        """
        Sample actions from the Gaussian policy.
        
        Args:
            policy_params: Output from forward() containing means and stds
            deterministic: If True, return means; if False, sample from distribution
        
        Returns:
            actions: (B, N, action_dim) - sampled actions
            log_probs: (B, N) - log probabilities of sampled actions
        """
        means = policy_params['means']  # (B, N, action_dim)
        stds = policy_params['stds']    # (B, N, action_dim)
        
        if deterministic:
            # Use mean actions (for evaluation)
            raw_actions = means
            log_probs = torch.zeros(means.shape[:-1], device=means.device)  # (B, N)
        else:
            # Sample from Gaussian distribution
            dist = torch.distributions.Normal(means, stds)
            raw_actions = dist.rsample()  # (B, N, action_dim) with gradients
            
            # Compute log probabilities
            log_probs = dist.log_prob(raw_actions).sum(dim=-1)  # (B, N)
        
        if self.use_tanh:
            # 4. Tanh-squashing for actuator limits
            squashed_actions = torch.tanh(raw_actions)  # ∈ (-1, 1)^2
            
            # 5. Scale to actual actuator ranges
            # v ∈ [0, v_max]: (tanh + 1) / 2 * v_max
            # ω ∈ [-ω_max, ω_max]: tanh * ω_max
            actions = torch.zeros_like(squashed_actions)
            actions[..., 0] = (squashed_actions[..., 0] + 1) / 2 * self.v_max  # v
            actions[..., 1] = squashed_actions[..., 1] * self.omega_max        # ω
            
            if not deterministic:
                # Adjust log probabilities for tanh transformation
                # log π(a) = log π(u) - Σ log(1 - tanh²(u))
                tanh_correction = torch.log(1 - squashed_actions**2 + 1e-8).sum(dim=-1)
                log_probs = log_probs - tanh_correction  # (B, N)
        else:
            actions = raw_actions
        
        return actions, log_probs
    
    def evaluate_actions(self, policy_params: dict, actions: torch.Tensor):
        """
        Evaluate log probabilities of given actions under current policy.
        Used for PPO policy updates.
        
        Args:
            policy_params: Current policy parameters
            actions: (B, N, action_dim) - actions to evaluate
        
        Returns:
            log_probs: (B, N) - log probabilities of given actions
            entropy: (B, N) - policy entropy for each agent
        """
        means = policy_params['means']  # (B, N, action_dim)
        stds = policy_params['stds']    # (B, N, action_dim)
        
        if self.use_tanh:
            # Inverse transform: from action space back to pre-tanh space
            # v: [0, v_max] → [-1, 1] → raw
            # ω: [-ω_max, ω_max] → [-1, 1] → raw
            normalized = torch.zeros_like(actions)
            normalized[..., 0] = 2 * actions[..., 0] / self.v_max - 1  # v
            normalized[..., 1] = actions[..., 1] / self.omega_max       # ω
            
            # Clamp to avoid numerical issues with atanh
            normalized = torch.clamp(normalized, min=-0.999, max=0.999)
            raw_actions = torch.atanh(normalized)
        else:
            raw_actions = actions
        
        # Compute log probabilities
        dist = torch.distributions.Normal(means, stds)
        log_probs = dist.log_prob(raw_actions).sum(dim=-1)  # (B, N)
        
        if self.use_tanh:
            # Adjust for tanh transformation
            tanh_correction = torch.log(1 - normalized**2 + 1e-8).sum(dim=-1)
            log_probs = log_probs - tanh_correction
        
        # Compute entropy (before tanh transformation)
        entropy = dist.entropy().sum(dim=-1)  # (B, N)
        
        return log_probs, entropy


class CriticHead(nn.Module):
    """
    Critic Head for centralized value function estimation.
    
    Takes the global context vector z_t ∈ R^(2D) (mean-pooled output from RSA) 
    and computes an approximation of the expected return (discounted sum of 
    future rewards) from that state under the current policy:
    
    V_φ(s_t) ≈ E_π_θ[Σ_{k=0}^∞ γ^k r_{t+k} | s_t]
    
    Mathematical specification:
    1. Input: z_t ∈ R^(2D) (2D=320 for concatenated RSA output)
    2. MLP: z_t → h_t^(1) = ReLU(W^(1) z_t + b^(1)) ∈ R^H
    3. Output: V_φ(s_t) = W^(2) h_t^(1) + b^(2) ∈ R (scalar)
    
    This is used in PPO for advantage computation and value loss minimization.
    """
    
    def __init__(self, rsa_output_dim: int = 320, hidden_dim: int = 64):
        """
        Args:
            rsa_output_dim: Input dimension (2D=320 from RSA global context)
            hidden_dim: Hidden layer size H (64 as suggested)
        """
        super().__init__()
        self.rsa_output_dim = rsa_output_dim
        self.hidden_dim = hidden_dim
        
        # Simple MLP: 2D → H → 1
        self.mlp = nn.Sequential(
            nn.Linear(rsa_output_dim, hidden_dim),   # W^(1) ∈ R^{H×2D}
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)                 # W^(2) ∈ R^{1×H} → scalar
        )
        
        # Initialize weights for stable training
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with reasonable values for value function learning"""
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                # Orthogonal initialization for stable training
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)
        
        # Final layer: smaller initialization for value head
        nn.init.orthogonal_(self.mlp[-1].weight, gain=0.1)
        nn.init.constant_(self.mlp[-1].bias, 0.0)
    
    def forward(self, global_context: torch.Tensor) -> torch.Tensor:
        """
        Estimate value function for the current state.
        
        Args:
            global_context: (B, 2D) - z_t from RSA block, represents global state
        
        Returns:
            values: (B,) - V_φ(s_t) scalar value estimates for each batch sample
        """
        B, rsa_dim = global_context.shape
        assert rsa_dim == self.rsa_output_dim, f"Expected rsa_output_dim={self.rsa_output_dim}, got {rsa_dim}"
        
        # MLP forward pass: z_t → V_φ(s_t)
        value_logits = self.mlp(global_context)  # (B, 1)
        values = value_logits.squeeze(-1)        # (B,) - remove last dimension
        
        return values


# Complete Multi-Agent Network combining all components
class MultiAgentNetwork(nn.Module):
    """
    Complete multi-agent network combining all components:
    ObservationEncoder → CopyConcat → Concatenate → RSABlock → ActorHead + CriticHead
                                     ↗                       ↙              ↘
                               Original + Teammate      Actions         Value Function
                                Embeddings           (Decentralized)    (Centralized)
    
    This implements the full CTDE (Centralized Training, Decentralized Execution)
    architecture from the paper for your 2-defender AUV scenario, with the fix
    to preserve self-information by concatenating original and teammate-aware embeddings.
    """
    
    def __init__(self, 
                 obs_dim_per_defender: int = 53,
                 n_agents: int = 2,
                 d_enc: int = 160,  # Total embedding dimension from 3-branch ObservationEncoder
                 num_heads: int = 8,  # 8 works with 320-dim input (160*2)
                 v_max: float = 5.0,
                 omega_max: float = 0.5):
        """
        Args:
            obs_dim_per_defender: Observation dimension per agent (53 for your scenario)
            n_agents: Number of agents (2 defenders)
            d_enc: Total embedding dimension (160 from 3-branch ObservationEncoder: 64+32+64)
            num_heads: Number of attention heads in RSA (8 works with 320-dim input)
            v_max: Maximum velocity constraint
            omega_max: Maximum angular velocity constraint
        """
        super().__init__()
        self.obs_dim_per_defender = obs_dim_per_defender
        self.n_agents = n_agents
        self.d_enc = d_enc
        self.rsa_input_dim = 2 * d_enc  # Concatenated: original + copyconcat
        
        # Network components
        self.observation_encoder = ObservationEncoder(
            obs_dim_per_defender=obs_dim_per_defender,
            max_neighbors=n_agents-1,
            max_attackers=1,
            d_self=64,
            d_team=32,
            d_env=64  # Total: 64+32+64 = 160 dims
        )
        
        self.copy_concat = CopyConcat(
            n_agents=n_agents,
            emb_dim=d_enc  # Processes 160-dim embeddings
        )
        
        self.rsa_block = RSABlock(
            input_dim=self.rsa_input_dim,  # 320 = 160 (original) + 160 (copyconcat)
            num_heads=num_heads
        )
        
        self.actor_head = ActorHead(
            rsa_output_dim=self.rsa_input_dim,  # 320
            v_max=v_max,
            omega_max=omega_max
        )
        
        self.critic_head = CriticHead(
            rsa_output_dim=self.rsa_input_dim  # 320
        )
    
    def forward(self, observations: torch.Tensor, deterministic: bool = False):
        """
        Complete forward pass through the multi-agent network.
        
        Args:
            observations: (B, N*obs_dim) - flattened observations from environment
                         For your scenario: (B, 2*53) = (B, 106)
            deterministic: Whether to use deterministic policy (for evaluation)
        
        Returns:
            dict containing:
                - 'actions': (B, N, 2) - sampled actions for each agent
                - 'action_log_probs': (B, N) - log probabilities of actions
                - 'values': (B,) - value function estimates
                - 'policy_params': dict with means and stds for each agent
        """
        B = observations.shape[0]
        
        # Reshape observations: (B, N*obs_dim) → (B, N, obs_dim)
        obs_reshaped = observations.view(B, self.n_agents, self.obs_dim_per_defender)
        
        # 1. Encode each agent's observation: õ_i = [f_self || f_team || f_env]
        embeddings = []
        for i in range(self.n_agents):
            agent_obs = obs_reshaped[:, i, :]  # (B, obs_dim)
            agent_emb = self.observation_encoder(agent_obs)  # (B, d_enc) = (B, 160)
            embeddings.append(agent_emb)
        
        # Stack embeddings: list of (B, d_enc) → (B, N, d_enc)
        original_embeddings = torch.stack(embeddings, dim=1)  # (B, N, 160)
        
        # 2. Copy-and-Concat: teammate information fusion
        # Takes all agents' embeddings and creates teammate-aware versions
        teammate_aware_embeddings = self.copy_concat(original_embeddings)  # (B, N, d_enc)
        
        # 3. Concatenate original + teammate-aware embeddings
        # original: [f_self||f_team||f_env], teammate_aware: cross-agent info via copy-concat
        final_embeddings = torch.cat([
            original_embeddings,           # (B, N, d_enc) - own [self+team+env] embedding
            teammate_aware_embeddings      # (B, N, d_enc) - other agents' embedding info  
        ], dim=-1)  # (B, N, 2*d_enc) = (B, N, 320)
        
        # 4. RSA: multi-head self-attention communication on concatenated embeddings
        refined_embeddings, global_context = self.rsa_block(final_embeddings)
        # refined_embeddings: (B, N, 2*d_enc) - for actors
        # global_context: (B, 2*d_enc) - for critic
        
        # 5. Actor Head: generate policies
        policy_params = self.actor_head(refined_embeddings, global_context)
        actions, action_log_probs = self.actor_head.sample_actions(
            policy_params, deterministic=deterministic
        )
        
        # 6. Critic Head: estimate values
        values = self.critic_head(global_context)  # (B,)
        
        return {
            'actions': actions,                    # (B, N, 2)
            'action_log_probs': action_log_probs, # (B, N)
            'values': values,                     # (B,)
            'policy_params': policy_params,       # dict with means/stds
            'global_context': global_context      # (B, 2*d_enc) for debugging
        }












