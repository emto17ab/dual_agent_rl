import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Dirichlet, Beta
from torch_geometric.nn import GCNConv

class GNNCritic(nn.Module):
    """
    Critic parametrizing the value function estimator V(s_t).
    """
    def __init__(self, in_channels, hidden_size, act_dim):
        super().__init__()
        self.act_dim = act_dim
        self.in_channels = in_channels
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)
    
    def forward(self, data):
        out = F.relu(self.conv1(data.x, data.edge_index))
        x = out + data.x
        x = x.reshape(-1, self.act_dim, self.in_channels)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = torch.sum(x, dim=1)  # Shape: [1, hidden_size]
        x = self.lin3(x)  # Shape: [1, 1]
        x = x.squeeze(0)  # Shape: [1] - consistent scalar output
        return x

class GNNActor(nn.Module):
    def __init__(self, in_channels, hidden_size, act_dim, mode, od_price_actions=False):
        super().__init__()
        self.mode = mode
        self.in_channels = in_channels
        self.act_dim = act_dim
        self.od_price_actions = od_price_actions
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        
        # Calculate output dimensions based on mode and OD pricing
        if mode == 0:
            # Mode 0: rebalancing only (no pricing)
            self.lin3 = nn.Linear(hidden_size, 1)
        elif mode == 1:
            # Mode 1: pricing only
            if od_price_actions:
                # OD-based pricing: output N×N price scalars per region
                # Each region outputs N scalars (one per destination)
                self.lin3 = nn.Linear(hidden_size, 2 * act_dim)  # 2 Beta params × N destinations
            else:
                # Origin-based pricing: output 1 price scalar per region
                self.lin3 = nn.Linear(hidden_size, 2)  # 2 Beta params
        else:
            # Mode 2: pricing + rebalancing
            if od_price_actions:
                # OD-based pricing + rebalancing
                self.lin3 = nn.Linear(hidden_size, 2 * act_dim + 1)  # (2 Beta params × N destinations) + 1 Dirichlet param
            else:
                # Origin-based pricing + rebalancing
                self.lin3 = nn.Linear(hidden_size, 3)  # 2 Beta params + 1 Dirichlet param

    def forward(self, data, deterministic=False):
        out = F.relu(self.conv1(data.x, data.edge_index))
        x = out + data.x
        x = x.reshape(-1, self.act_dim, self.in_channels)
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        
        # Output concentration parameters
        # Use softplus for positivity (output >= 0), add small epsilon for numerical stability
        x = F.softplus(self.lin3(x))
        
        # Handle concentration parameters based on mode and OD pricing
        if self.mode == 0:
            # Mode 0: Dirichlet - squeeze last dim to get [1, nregion]
            assert x.shape == (1, self.act_dim, 1), f"Mode 0: Expected shape (1, {self.act_dim}, 1), got {x.shape}"
            concentration = x.squeeze(-1) + 0.1
            assert concentration.shape == (1, self.act_dim), f"Mode 0: Expected concentration shape (1, {self.act_dim}), got {concentration.shape}"
        elif self.mode == 1:
            if self.od_price_actions:
                # Mode 1 OD: Beta parameters for each OD pair
                # x shape: [1, nregion, 2*nregion]
                assert x.shape == (1, self.act_dim, 2 * self.act_dim), f"Mode 1 OD: Expected shape (1, {self.act_dim}, {2*self.act_dim}), got {x.shape}"
                # Reshape to [1, nregion, nregion, 2] (origin, destination, beta_params)
                concentration = x.view(1, self.act_dim, self.act_dim, 2) + 1
            else:
                # Mode 1 origin: Beta - keep [1, nregion, 2]
                assert x.shape == (1, self.act_dim, 2), f"Mode 1: Expected shape (1, {self.act_dim}, 2), got {x.shape}"
                concentration = x + 1
        else:
            if self.od_price_actions:
                # Mode 2 OD: Beta for pricing + Dirichlet for rebalancing
                # x shape: [1, nregion, 2*nregion + 1]
                assert x.shape == (1, self.act_dim, 2 * self.act_dim + 1), f"Mode 2 OD: Expected shape (1, {self.act_dim}, {2*self.act_dim + 1}), got {x.shape}"
                concentration = x.clone()
                # Reshape Beta parameters to [1, nregion, nregion, 2]
                beta_params = concentration[:, :, :2*self.act_dim].view(1, self.act_dim, self.act_dim, 2) + 1
                # Keep Dirichlet parameter [1, nregion, 1]
                dirichlet_param = concentration[:, :, 2*self.act_dim:] + 0.1
                # Concatenate: [1, nregion, nregion, 2] and [1, nregion, 1]
                concentration = {'beta': beta_params, 'dirichlet': dirichlet_param}
            else:
                # Mode 2 origin: Beta + Dirichlet - keep [1, nregion, 3]
                assert x.shape == (1, self.act_dim, 3), f"Mode 2: Expected shape (1, {self.act_dim}, 3), got {x.shape}"
                concentration = x.clone()
                concentration[:,:,:2] = concentration[:,:,:2] + 1  # Add 1 to Beta parameters
                concentration[:,:,2] = concentration[:,:,2] + 0.1  # Add 0.1 to Dirichlet parameters
    
        if deterministic:
            if self.mode == 0:
                # Dirichlet mean: normalize concentration parameters
                # Shape: [1, nregion] -> [nregion]
                action = concentration / concentration.sum()
                action = action.squeeze(0)
            elif self.mode == 1:
                if self.od_price_actions:
                    # OD Beta mean: alpha / (alpha + beta)
                    # Shape: [1, nregion, nregion, 2] -> [nregion, nregion]
                    action_o = concentration[:,:,:,0] / (concentration[:,:,:,0] + concentration[:,:,:,1])
                    action_o[action_o<0] = 0
                    action = action_o.squeeze(0)
                else:
                    # Origin Beta mean: alpha / (alpha + beta)
                    # Shape: [1, nregion, 2] -> [nregion]
                    action_o = concentration[:,:,0] / (concentration[:,:,0] + concentration[:,:,1])
                    action_o[action_o<0] = 0
                    action = action_o.squeeze(0)
            else:
                if self.od_price_actions:
                    # Mode 2 OD: Beta mean for pricing + Dirichlet mean for rebalancing
                    # Pricing shape: [1, nregion, nregion, 2] -> [nregion, nregion]
                    beta_conc = concentration['beta']
                    action_o = beta_conc[:,:,:,0] / (beta_conc[:,:,:,0] + beta_conc[:,:,:,1])
                    action_o[action_o<0] = 0
                    # Rebalancing shape: [1, nregion, 1] -> [nregion]
                    dirichlet_conc = concentration['dirichlet']
                    action_reb = dirichlet_conc.squeeze(-1) / dirichlet_conc.squeeze(-1).sum()
                    # Combined shape: [nregion, nregion, 2] (price OD matrix + rebalancing vector stacked)
                    # Stack as [nregion, nregion+1] where first nregion cols are OD prices, last col is rebalancing
                    action = torch.cat((action_o.squeeze(0), action_reb.squeeze(0).unsqueeze(-1)), dim=-1)
                else:
                    # Mode 2 origin: Beta mean for pricing + Dirichlet mean for rebalancing
                    # Pricing shape: [1, nregion] -> [nregion]
                    action_o = concentration[:,:,0] / (concentration[:,:,0] + concentration[:,:,1])
                    action_o[action_o<0] = 0
                    # Rebalancing shape: [1, nregion] -> [nregion]
                    action_reb = concentration[:,:,2] / concentration[:,:,2].sum()
                    # Combined shape: [nregion, 2]
                    action = torch.stack((action_o.squeeze(0), action_reb.squeeze(0)), dim=-1)
            log_prob = None
        else:
            if self.mode == 0:
                # Dirichlet: single joint distribution over nregion categories
                # concentration shape: [1, nregion]
                m = Dirichlet(concentration)
                action = m.rsample()  # Shape: [1, nregion]
                log_prob = m.log_prob(action)  # Shape: [1] (scalar for joint distribution)
                action = action.squeeze(0)  # Shape: [nregion]
                assert action.shape == (self.act_dim,), f"Mode 0: Expected action shape ({self.act_dim},), got {action.shape}"
                assert log_prob.shape == (1,), f"Mode 0: Expected log_prob shape (1,), got {log_prob.shape}"
            elif self.mode == 1:
                if self.od_price_actions:
                    # OD Beta: nregion × nregion independent distributions
                    # concentration shape: [1, nregion, nregion, 2]
                    m_o = Beta(concentration[:,:,:,0], concentration[:,:,:,1])
                    action_o = m_o.rsample()  # Shape: [1, nregion, nregion]
                    # Sum log probs across all OD pairs
                    log_prob = m_o.log_prob(action_o).sum(dim=[1, 2])  # Shape: [1]
                    action = action_o.squeeze(0)  # Shape: [nregion, nregion]
                    assert action.shape == (self.act_dim, self.act_dim), f"Mode 1 OD: Expected action shape ({self.act_dim}, {self.act_dim}), got {action.shape}"
                    assert log_prob.shape == (1,), f"Mode 1 OD: Expected log_prob shape (1,), got {log_prob.shape}"
                else:
                    # Origin Beta: nregion independent distributions
                    # concentration shape: [1, nregion, 2]
                    m_o = Beta(concentration[:,:,0], concentration[:,:,1])
                    action_o = m_o.rsample()  # Shape: [1, nregion]
                    # Sum log probs across independent distributions (not mean!)
                    log_prob = m_o.log_prob(action_o).sum(dim=-1)  # Shape: [1]
                    action = action_o.squeeze(0)  # Shape: [nregion]
                    assert action.shape == (self.act_dim,), f"Mode 1: Expected action shape ({self.act_dim},), got {action.shape}"
                    assert log_prob.shape == (1,), f"Mode 1: Expected log_prob shape (1,), got {log_prob.shape}"
            else:
                if self.od_price_actions:
                    # Mode 2 OD: nregion × nregion independent Beta + 1 Dirichlet
                    beta_conc = concentration['beta']
                    dirichlet_conc = concentration['dirichlet']
                    # Beta for OD pricing
                    m_o = Beta(beta_conc[:,:,:,0], beta_conc[:,:,:,1])
                    action_o = m_o.rsample()  # Shape: [1, nregion, nregion]
                    # Dirichlet for rebalancing
                    m_reb = Dirichlet(dirichlet_conc.squeeze(-1))
                    action_reb = m_reb.rsample()  # Shape: [1, nregion]
                    # Joint log prob: sum of all Beta log probs + Dirichlet log prob
                    log_prob = m_o.log_prob(action_o).sum(dim=[1, 2]) + m_reb.log_prob(action_reb)  # Shape: [1]
                    # Combined action shape: [nregion, nregion+1]
                    action = torch.cat((action_o.squeeze(0), action_reb.squeeze(0).unsqueeze(-1)), dim=-1)
                    assert action.shape == (self.act_dim, self.act_dim + 1), f"Mode 2 OD: Expected action shape ({self.act_dim}, {self.act_dim + 1}), got {action.shape}"
                    assert log_prob.shape == (1,), f"Mode 2 OD: Expected log_prob shape (1,), got {log_prob.shape}"
                else:
                    # Mode 2 origin: nregion independent Beta distributions + 1 Dirichlet
                    # Beta concentration shape: [1, nregion, 2]
                    m_o = Beta(concentration[:,:,0], concentration[:,:,1])
                    action_o = m_o.rsample()  # Shape: [1, nregion]
                    # Dirichlet for rebalancing
                    # Rebalancing concentration shape: [1, nregion]
                    m_reb = Dirichlet(concentration[:,:,2])
                    action_reb = m_reb.rsample()  # Shape: [1, nregion]
                    # Joint log prob: sum of Beta log probs + Dirichlet log prob
                    log_prob = m_o.log_prob(action_o).sum(dim=-1) + m_reb.log_prob(action_reb)  # Shape: [1]
                    # Combined action shape: [nregion, 2]
                    action = torch.stack((action_o.squeeze(0), action_reb.squeeze(0)), dim=-1)
                    assert action.shape == (self.act_dim, 2), f"Mode 2: Expected action shape ({self.act_dim}, 2), got {action.shape}"
                    assert log_prob.shape == (1,), f"Mode 2: Expected log_prob shape (1,), got {log_prob.shape}"
            
        return action, log_prob, concentration