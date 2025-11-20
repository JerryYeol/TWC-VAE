import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .util import load_h5ad_to_dataloader

class Encoderc(nn.Module):
    def __init__(self, input_dim, z_dim, dropout_rate):
        super(Encoderc, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_dim, 800),
                                 nn.BatchNorm1d(800, momentum=0.01, eps=0.001),
                                  nn.LayerNorm(800, elementwise_affine=False),
                                 nn.LeakyReLU(),
                                 nn.Dropout(p=dropout_rate))
        self.fc2 = nn.Sequential(nn.Linear(800, 800),
                                 nn.BatchNorm1d(800, momentum=0.01, eps=0.001),
                                  nn.LayerNorm(800, elementwise_affine=False),
                                 nn.LeakyReLU(),
                                 nn.Dropout(p=dropout_rate))
        # self.fc3 = nn.Sequential(nn.Linear(512, 256),
        #                          nn.BatchNorm1d(256),
        #                          nn.ReLU())
        # self.fc_mean = nn.Sequential(nn.Linear(800, z_dim),
        #                              nn.ReLU())
        # self.fc_log_var = nn.Sequential(nn.Linear(800, z_dim),
        #                              nn.ReLU())
        self.fc_mean = nn.Linear(800, z_dim)
        self.fc_log_var = nn.Linear(800, z_dim)
        
    def forward(self, x):
        h = self.fc1(x)
        h = self.fc2(h)
        # h = self.fc3(h)
        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        return mean, log_var
    
class Encoderp(nn.Module):
    def __init__(self, input_dim, z_dim, dropout_rate):
        super(Encoderp, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_dim, 800),
                                 nn.BatchNorm1d(800, momentum=0.01, eps=0.001),
                                  nn.LayerNorm(800, elementwise_affine=False),
                                 nn.LeakyReLU(),
                                 nn.Dropout(p=dropout_rate))
        self.fc2 = nn.Sequential(nn.Linear(800, 800),
                                 nn.BatchNorm1d(800, momentum=0.01, eps=0.001),
                                  nn.LayerNorm(800, elementwise_affine=False),
                                 nn.LeakyReLU(),
                                 nn.Dropout(p=dropout_rate))
        # self.fc3 = nn.Sequential(nn.Linear(512, 256),
        #                          nn.BatchNorm1d(256),
        #                          nn.ReLU())
        # self.fc_mean = nn.Sequential(nn.Linear(800, z_dim),
        #                              nn.ReLU())
        # self.fc_log_var = nn.Sequential(nn.Linear(800, z_dim),
        #                              nn.ReLU())
        self.fc_mean = nn.Linear(800, z_dim)
        self.fc_log_var = nn.Linear(800, z_dim)
        
    def forward(self, x):
        h = self.fc1(x)
        h = self.fc2(h)
        # h = self.fc3(h)
        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        return mean, log_var
    
class Decoderc(nn.Module):
    def __init__(self, z_dim, output_dim, dropout_rate):
        super(Decoderc, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(z_dim, 800),
                                 # nn.BatchNorm1d(800, momentum=0.01, eps=0.001),
                                 # nn.LayerNorm(800, elementwise_affine=False),
                                 nn.LeakyReLU(),
                                 nn.Dropout(p=dropout_rate))
        self.fc2 = nn.Sequential(nn.Linear(800, 800),
                                 # nn.BatchNorm1d(800, momentum=0.01, eps=0.001),
                                 # nn.LayerNorm(800, elementwise_affine=False),
                                 nn.LeakyReLU(),
                                 nn.Dropout(p=dropout_rate))
        self.fc3 = nn.Sequential(nn.Linear(800, output_dim),
                                  nn.ReLU())
        # self.fc3 = nn.Linear(800, output_dim)
        
    def forward(self, z):
        h = self.fc1(z)
        h = self.fc2(h)
        h = self.fc3(h)
        return h

class Decoderp(nn.Module):
    def __init__(self, z_dim, output_dim, dropout_rate):
        super(Decoderp, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(z_dim, 800),
                                 # nn.BatchNorm1d(800, momentum=0.01, eps=0.001),
                                 # nn.LayerNorm(800, elementwise_affine=False),
                                 nn.LeakyReLU(),
                                 nn.Dropout(p=dropout_rate))
        self.fc2 = nn.Sequential(nn.Linear(800, 800),
                                 # nn.BatchNorm1d(800, momentum=0.01, eps=0.001),
                                 # nn.LayerNorm(800, elementwise_affine=False),
                                 nn.LeakyReLU(),
                                 nn.Dropout(p=dropout_rate))
        self.fc3 = nn.Sequential(nn.Linear(800, output_dim),
                                  nn.ReLU())
        # self.fc3 = nn.Linear(800, output_dim)
        
    def forward(self, z):
        h = self.fc1(z)
        h = self.fc2(h)
        h = self.fc3(h)
        return h


class Decodercp(nn.Module):
    def __init__(self, z_dim, output_dim, dropout_rate):
        super(Decodercp, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(z_dim, 800),
                                 # nn.BatchNorm1d(800, momentum=0.01, eps=0.001),
                                 # nn.LayerNorm(800, elementwise_affine=False),
                                 nn.LeakyReLU(),
                                 nn.Dropout(p=dropout_rate))
        self.fc2 = nn.Sequential(nn.Linear(800, 800),
                                 # nn.BatchNorm1d(800, momentum=0.01, eps=0.001),
                                 # nn.LayerNorm(800, elementwise_affine=False),
                                 nn.LeakyReLU(),
                                 nn.Dropout(p=dropout_rate))
        self.fc3 = nn.Sequential(nn.Linear(800, output_dim),
                                  nn.ReLU())
        # self.fc3 = nn.Linear(800, output_dim)
        
    def forward(self, z):
        h = self.fc1(z)
        h = self.fc2(h)
        h = self.fc3(h)
        return h


class Decoderpc(nn.Module):
    def __init__(self, z_dim, output_dim, dropout_rate):
        super(Decoderpc, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(z_dim, 800),
                                 # nn.BatchNorm1d(800, momentum=0.01, eps=0.001),
                                 # nn.LayerNorm(800, elementwise_affine=False),
                                 nn.LeakyReLU(),
                                 nn.Dropout(p=dropout_rate))
        self.fc2 = nn.Sequential(nn.Linear(800, 800),
                                 # nn.BatchNorm1d(800, momentum=0.01, eps=0.001),
                                 # nn.LayerNorm(800, elementwise_affine=False),
                                 nn.LeakyReLU(),
                                 nn.Dropout(p=dropout_rate))
        self.fc3 = nn.Sequential(nn.Linear(800, output_dim),
                                  nn.ReLU())
        # self.fc3 = nn.Linear(800, output_dim)
        
    def forward(self, z):
        h = self.fc1(z)
        h = self.fc2(h)
        h = self.fc3(h)
        return h


class Couplerc(nn.Module):
    def __init__(self, z_dim):
        super(Couplerc, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(z_dim, z_dim),
                                 nn.BatchNorm1d(z_dim),
                                 nn.ReLU())
        self.fc_mean = nn.Linear(z_dim, z_dim)
        self.fc_log_var = nn.Linear(z_dim, z_dim)
        
    def forward(self, z):
        h = self.fc1(z)
        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        return mean, log_var
    
class Couplerp(nn.Module):
    def __init__(self, z_dim):
        super(Couplerp, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(z_dim, z_dim),
                                 nn.BatchNorm1d(z_dim),
                                 nn.ReLU())
        self.fc_mean = nn.Linear(z_dim, z_dim)
        self.fc_log_var = nn.Linear(z_dim, z_dim)
        
    def forward(self, z):
        h = self.fc1(z)
        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        return mean, log_var
    
# class CrossAttention(nn.Module):
#     def __init__(self, z_dim, n_heads=4, dropout=0.1):
#         super(CrossAttention, self).__init__()
#         self.attn = nn.MultiheadAttention(embed_dim=z_dim, num_heads=n_heads, dropout=dropout, batch_first=True)

#     def forward(self, query, key, value):
#         # query, key, value: shape [batch_size, seq_len, z_dim]
#         attn_output, _ = self.attn(query, key, value)
#         return attn_output
import torch
import torch.nn as nn
import math


# class CrossAttention(nn.Module):
#     def __init__(self, z_dim, down_dim=128, up_dim=256, num_heads=8, rope_head_dim=26, dropout_prob=0.1):
#         super(CrossAttention, self).__init__()

#         self.z_dim = z_dim
#         self.down_dim = down_dim
#         self.up_dim = up_dim
#         self.num_heads = num_heads
#         self.head_dim = z_dim // num_heads
#         self.rope_head_dim = rope_head_dim
#         self.v_head_dim = up_dim // num_heads

#         # Low-rank projections for key, value, and query
#         self.down_proj_kv = nn.Linear(z_dim, down_dim)  # W^{DKV}
#         self.up_proj_k = nn.Linear(down_dim, up_dim)  # W^{UK}
#         self.up_proj_v = nn.Linear(down_dim, up_dim)  # W^{UV}
#         self.down_proj_q = nn.Linear(z_dim, down_dim)  # W^{DQ}
#         self.up_proj_q = nn.Linear(down_dim, up_dim)  # W^{UQ}

#         # Decoupled projections for MQA (Multi-Query Attention)
#         self.proj_qr = nn.Linear(down_dim, rope_head_dim * num_heads)
#         self.proj_kr = nn.Linear(z_dim, rope_head_dim * 1)

#         # Rotary embeddings for query and key
#         self.rope_q = RotaryEmbedding(rope_head_dim * num_heads, num_heads)
#         self.rope_k = RotaryEmbedding(rope_head_dim, 1)

#         # Dropout and final linear layer
#         self.dropout = nn.Dropout(dropout_prob)
#         self.fc = nn.Linear(num_heads * self.v_head_dim, z_dim)
#         self.res_dropout = nn.Dropout(dropout_prob)

#     def forward(self, h, mask=None):
#         bs, seq_len, _ = h.size()

#         # Step 1: Low-rank transformation
#         c_t_kv = self.down_proj_kv(h)
#         k_t_c = self.up_proj_k(c_t_kv)
#         v_t_c = self.up_proj_v(c_t_kv)
#         c_t_q = self.down_proj_q(h)
#         q_t_c = self.up_proj_q(c_t_q)

#         # Step 2: Decoupled projections for MQA, with ROPE
#         q_t_r = self.rope_q(self.proj_qr(c_t_q))
#         k_t_r = self.rope_k(self.proj_kr(h))

#         # Step 3: Combine the results of Step 1 and Step 2
#         q_t_c = q_t_c.reshape(bs, seq_len, self.num_heads, -1).transpose(1, 2)
#         q = torch.cat([q_t_c, q_t_r], dim=-1)

#         k_t_c = k_t_c.reshape(bs, seq_len, self.num_heads, -1).transpose(1, 2)
#         k_t_r = k_t_r.repeat(1, self.num_heads, 1, 1)
#         k = torch.cat([k_t_c, k_t_r], dim=-1)

#         # Attention computation: [bs, num_heads, seq_len, seq_len]
#         scores = torch.matmul(q, k.transpose(-1, -2))
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, -1e9)
#         scores = torch.softmax(scores / (math.sqrt(self.head_dim) + math.sqrt(self.rope_head_dim)), dim=-1)
#         scores = self.dropout(scores)

#         # Perform attention computation with v_t_c and scores
#         v_t_c = v_t_c.reshape(bs, seq_len, self.num_heads, self.v_head_dim).transpose(1, 2)
#         output = torch.matmul(scores, v_t_c)

#         # Final linear transformation
#         output = output.transpose(1, 2).reshape(bs, seq_len, -1)
#         output = self.fc(output)
#         output = self.res_dropout(output)

#         return output


class RotaryEmbedding(nn.Module):
    def __init__(self, d_model, num_heads, base=10000, max_len=512):
        super().__init__()
        self.head_dim = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.base = base
        self.max_len = max_len
        # Initialize the position embeddings
        self.cos_pos_cache, self.sin_pos_cache = self._compute_pos_emb()

    def _compute_pos_emb(self):
        # theta_i = 1 / (10000^(2i/head_dim)) for i in [0, head_dim//2]
        theta_i = 1. / (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        # Create position indices for each position in the sequence
        positions = torch.arange(self.max_len)
        # Apply the scaling factor to get position embeddings
        pos_emb = positions.unsqueeze(1) * theta_i.unsqueeze(0)
        # Calculate the sin and cos embeddings
        cos_pos = pos_emb.sin().repeat_interleave(2, dim=-1)
        sin_pos = pos_emb.cos().repeat_interleave(2, dim=-1)

        return cos_pos, sin_pos

    def forward(self, q):
        bs, q_len = q.shape[0], q.shape[1]
        
        # Ensure that cos_pos and sin_pos are on the same device as q
        device = q.device  # Get the device of the input tensor (usually CUDA)
        self.cos_pos = self.cos_pos_cache[:q_len].to(device)
        self.sin_pos = self.sin_pos_cache[:q_len].to(device)

        # Reshape query tensor to apply ROPE for each head
        q = q.reshape(bs, q_len, self.num_heads, -1).transpose(1, 2)

        # Repeat the position embeddings across the batch and head dimensions
        self.cos_pos = self.cos_pos.repeat(bs, self.num_heads, 1, 1)
        self.sin_pos = self.sin_pos.repeat(bs, self.num_heads, 1, 1)

        # Apply ROPE by interleaving positive and negative parts of the query
        q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
        q2 = q2.reshape(bs, self.num_heads, q_len, -1)

        # Apply the sin and cos position embeddings to the query tensor
        r_q = q * self.cos_pos + q2 * self.sin_pos

        return r_q
class CrossAttention(nn.Module):
    def __init__(self, z_dim, down_dim=128, up_dim=256, num_heads=8, rope_head_dim=26, dropout_prob=0.1):
        super(CrossAttention, self).__init__()

        self.z_dim = z_dim
        self.down_dim = down_dim
        self.up_dim = up_dim
        self.num_heads = num_heads
        self.head_dim = z_dim // num_heads
        self.rope_head_dim = rope_head_dim
        self.v_head_dim = up_dim // num_heads

        # Low-rank projections for key, value, and query
        self.down_proj_kv = nn.Linear(z_dim, down_dim)  # W^{DKV}
        self.up_proj_k = nn.Linear(down_dim, up_dim)  # W^{UK}
        self.up_proj_v = nn.Linear(down_dim, up_dim)  # W^{UV}
        self.down_proj_q = nn.Linear(z_dim, down_dim)  # W^{DQ}
        self.up_proj_q = nn.Linear(down_dim, up_dim)  # W^{UQ}

        # Decoupled projections for MQA (Multi-Query Attention)
        self.proj_qr = nn.Linear(down_dim, rope_head_dim * num_heads)
        self.proj_kr = nn.Linear(z_dim, rope_head_dim * 1)

        # Rotary embeddings for query and key
        self.rope_q = RotaryEmbedding(rope_head_dim * num_heads, num_heads)
        self.rope_k = RotaryEmbedding(rope_head_dim, 1)

        # Dropout and final linear layer
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(num_heads * self.v_head_dim, z_dim)
        self.res_dropout = nn.Dropout(dropout_prob)

    def forward(self, query, key, value, mask=None):
        bs, seq_len, _ = query.size()

        # Step 1: Low-rank transformation
        c_t_kv = self.down_proj_kv(key)
        k_t_c = self.up_proj_k(c_t_kv)
        v_t_c = self.up_proj_v(c_t_kv)
        c_t_q = self.down_proj_q(query)
        q_t_c = self.up_proj_q(c_t_q)

        # Step 2: Decoupled projections for MQA, with ROPE
        q_t_r = self.rope_q(self.proj_qr(c_t_q))
        k_t_r = self.rope_k(self.proj_kr(key))

        # Step 3: Combine the results of Step 1 and Step 2
        q_t_c = q_t_c.reshape(bs, seq_len, self.num_heads, -1).transpose(1, 2)
        q = torch.cat([q_t_c, q_t_r], dim=-1)

        k_t_c = k_t_c.reshape(bs, seq_len, self.num_heads, -1).transpose(1, 2)
        k_t_r = k_t_r.repeat(1, self.num_heads, 1, 1)
        k = torch.cat([k_t_c, k_t_r], dim=-1)

        # Attention computation: [bs, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-1, -2))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = torch.softmax(scores / (math.sqrt(self.head_dim) + math.sqrt(self.rope_head_dim)), dim=-1)
        scores = self.dropout(scores)

        # Perform attention computation with v_t_c and scores
        v_t_c = v_t_c.reshape(bs, seq_len, self.num_heads, self.v_head_dim).transpose(1, 2)
        output = torch.matmul(scores, v_t_c)

        # Final linear transformation
        output = output.transpose(1, 2).reshape(bs, seq_len, -1)
        output = self.fc(output)
        output = self.res_dropout(output)

        return output


class VAE(nn.Module):
    def __init__(self, x_dim, z_dim=16, learning_rate=0.001, dropout_rate=0.2, alpha=0.01, beta=1):
        super(VAE, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.alpha = alpha
        self.beta = beta
        
        self.encoder_c = Encoderc(x_dim, z_dim, dropout_rate)
        self.encoder_p = Encoderp(x_dim, z_dim, dropout_rate)
        
        self.decoder_c = Decoderc(z_dim, x_dim, dropout_rate)
        self.decoder_p = Decoderp(z_dim, x_dim, dropout_rate)
        self.decoder_cp = Decoderp(z_dim, x_dim, dropout_rate)
        self.decoder_pc = Decoderc(z_dim, x_dim, dropout_rate)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=z_dim,       # dimensionality of the model
            nhead=4,             # number of attention heads
            dim_feedforward=512, # optional, can be tuned
            dropout=dropout_rate
        )
        self.coupler_c = Couplerc(z_dim)
        self.coupler_p = Couplerp(z_dim)
        self.trans_c = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.trans_p = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.cross_cp = CrossAttention(z_dim)  # cross from c → p
        self.cross_pc = CrossAttention(z_dim)  # cross from p → c

    def sample_z(self, mean, log_var):
        eps = torch.randn(mean.size(0), self.z_dim).to(mean.device)
        return mean + torch.exp(log_var / 2) * eps

    def forward(self, x_0, x_1):
        mu_0, log_var_0 = self.encoder_c(x_0)
        mu_1, log_var_1 = self.encoder_p(x_1)
        
        z_mean_c = self.sample_z(mu_0, log_var_0)
        z_mean_p = self.sample_z(mu_1, log_var_1)

        z_trans_c = self.trans_c(z_mean_c)
        z_trans_p = self.trans_p(z_mean_p)
        # Add sequence dimension for cross-attention
        z_mean_c_seq = z_trans_c.unsqueeze(1)  # [batch, 1, z_dim]
        z_mean_p_seq = z_trans_p.unsqueeze(1)

        # Cross-attention: p attends to c, c attends to p
        attn_p_from_c = self.cross_cp(z_mean_p_seq, z_mean_c_seq, z_mean_c_seq)  # p attends to c
        attn_c_from_p = self.cross_pc(z_mean_c_seq, z_mean_p_seq, z_mean_p_seq)  # c attends to p

        # Optionally squeeze back
        attn_p_from_c = attn_p_from_c.squeeze(1)  # [batch, z_dim]
        attn_c_from_p = attn_c_from_p.squeeze(1)
        # Add or concatenate
        z_mean_p = z_mean_p + attn_p_from_c
        z_mean_c = z_mean_c + attn_c_from_p

        mu_p, log_var_p = self.coupler_c(z_mean_c)
        mu_c, log_var_c = self.coupler_p(z_mean_p)
        
        z_mean_1 = self.sample_z(mu_p, log_var_p)
        z_mean_0 = self.sample_z(mu_c, log_var_c)
        
        x_hat_0 = self.decoder_c(z_mean_c)
        x_hat_1 = self.decoder_p(z_mean_p)
        x_hat_cp = self.decoder_cp(z_mean_1)
        x_hat_pc = self.decoder_pc(z_mean_0)
        
        return x_hat_0, x_hat_1, x_hat_cp, x_hat_pc, z_mean_c, z_mean_p, z_mean_1, z_mean_0, mu_0, log_var_0, mu_1, log_var_1, mu_p, log_var_p, mu_c, log_var_c


    def loss_function(self, x_0, x_1, x_hat_0, x_hat_1, x_hat_cp, x_hat_pc, z_mean_c, z_mean_p, z_mean_1, z_mean_0, mu_0, log_var_0, mu_1, log_var_1, mu_p, log_var_p, mu_c, log_var_c):
# sigma = exp(0.5 * log_var)  —— 对角协方差的标准差
        sigma_0 = torch.exp(0.5 * log_var_0)
        sigma_1 = torch.exp(0.5 * log_var_1)

        # 2-Wasserstein 距离的平方（再乘 0.25）
        w2_loss0 = 0.25 * torch.sum(mu_0**2 + (sigma_0 - 1) ** 2, dim=1)
        w2_loss1 = 0.25 * torch.sum(mu_1**2 + (sigma_1 - 1) ** 2, dim=1)

        # 其余损失项保持不变
        recon_loss0 = 0.25 * torch.sum((x_0 - x_hat_0) ** 2, dim=1)
        trans_loss0 = 0.25 * torch.sum((x_0 - x_hat_pc) ** 2, dim=1)
        coupl_loss0 = 0.25 * torch.sum((mu_c - mu_0) ** 2, dim=1)

        recon_loss1 = 0.25 * torch.sum((x_1 - x_hat_1) ** 2, dim=1)
        trans_loss1 = 0.25 * torch.sum((x_1 - x_hat_cp) ** 2, dim=1)
        coupl_loss1 = 0.25 * torch.sum((mu_p - mu_1) ** 2, dim=1)
        
        kl_loss = w2_loss0 + w2_loss1
        recon_loss = recon_loss0 + recon_loss1
        trans_loss = trans_loss0 + trans_loss1
        coupl_loss = coupl_loss0 + coupl_loss1
        
        vae_loss = torch.mean(recon_loss + trans_loss + self.alpha * kl_loss + self.beta * coupl_loss)
        
        return vae_loss
    
    def to_latent(self, data):
        self.eval()
        adata = torch.tensor(data.X).float().to('cuda')
        
        
        dataset = TensorDataset(adata)
        dataloader = DataLoader(dataset, batch_size=data.X.shape[0], shuffle=True)
        with torch.no_grad():
            for data in dataloader:
                mu, log_var = self.encoder_c(data)
                latent = self.sample_z(mu, log_var)
                mu_p, log_var_p = self.coupler_c(latent)
                latent = self.sample_z(mu_p, log_var_p)    
        return latent
        
    def reconstruct(self, data):
        self.eval()
        
        with torch.no_grad():
            latent = self.to_latent(data)
            reconstruct = self.decoder_cp(latent)
        return reconstruct
        
    def predict(self, adata_c, adata_p, batch_size: int = 1024, device: str = "cuda") -> np.ndarray:
        """
        预测 adata_p 的条件下 adata_c 在另一模态上的表达（即 x_hat_cp）。
        """
        import numpy as np
        import scipy.sparse as sp

        self.eval()

        # ---------- 1. 拿出矩阵并转成 dense float32 ----------
        def _dense(x):
            if sp.issparse(x):
                x = x.toarray()
            if not isinstance(x, np.ndarray):
                x = np.asarray(x, dtype=np.float32)
            else:
                x = x.astype(np.float32, copy=False)
            return x

        x_c = _dense(adata_c.X if hasattr(adata_c, "X") else adata_c)
        x_p = _dense(adata_p.X if hasattr(adata_p, "X") else adata_p)

        # ---------- 2. NumPy → Tensor 搬到 device ----------
        t_c = torch.from_numpy(x_c).to(device)
        t_p = torch.from_numpy(x_p).to(device)
        min_len = min(t_c.shape[0], t_p.shape[0])
        if t_c.shape[0] != t_p.shape[0]:
            t_c = t_c[:min_len]
            t_p = t_p[:min_len]

        loader = DataLoader(
            TensorDataset(t_c, t_p),
            batch_size=batch_size,
            shuffle=False,     # 保证顺序不乱
            drop_last=False,
        )

        preds = []
        with torch.no_grad():
            for batch_c, batch_p in loader:
                _, _, x_hat_cp, _, *_ = self(batch_c, batch_p)  # 只关心 x_hat_cp
                preds.append(x_hat_cp.cpu())

        pred = torch.cat(preds, dim=0).numpy()  # (n_cells, x_dim)
        return pred
        

    def train_vae(self, adata_c, adata_p, batch_size=128, n_epochs=3, save_path=None, device='cuda'):
        self.train()
        self.to(device)
        train_data_c = torch.tensor(adata_c.X).float().to(device)
        train_data_p = torch.tensor(adata_p.X).float().to(device)
        
        dataset = TensorDataset(train_data_c, train_data_p)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        losses=[]
        
        for epoch in range(n_epochs):
            for data, data1 in dataloader:
                self.optimizer.zero_grad()
                x_hat_0, x_hat_1, x_hat_cp, x_hat_pc, z_mean_c, z_mean_p, z_mean_1, z_mean_0, mu_0, log_var_0, mu_1, log_var_1, mu_p, log_var_p, mu_c, log_var_c = self(data, data1)
                loss = self.loss_function(data, data1, x_hat_0, x_hat_1, x_hat_cp, x_hat_pc, z_mean_c, z_mean_p, z_mean_1, z_mean_0, mu_0, log_var_0, mu_1, log_var_1, mu_p, log_var_p, mu_c, log_var_c)
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch + 1}: Train VAE Loss: {loss.item()}")
            losses.append(loss.item())
        
        
        if save_path:
            torch.save(self.state_dict(), save_path)
            print(f"Model saved at {save_path}")

    def load_model(self, load_path):
        self.load_state_dict(torch.load(load_path))
        print(f"Model loaded from {load_path}")

