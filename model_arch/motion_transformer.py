'''
The code and architecture is adapted from MDM: https://github.com/GuyTevet/motion-diffusion-model
'''
import numpy as np
import torch
import torch.nn as nn
from model_arch.encoders import MLP_Encoder
import random


class MotionTransformer(nn.Module):
    def __init__(self, 
                 input_feats, 
                 latent_dim=256, 
                 ff_size=1024, 
                 num_layers=8, 
                 num_heads=4, 
                 dropout=0.1,
                 activation="gelu", 
                 bps_input_dim=3072, 
                 pred_horizon=32, 
                 contact_input_dim=1024, 
                 diffusion_step_embed_dim=256,
                 max_len=65):
        super().__init__()


        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.pred_horizon = pred_horizon 
        self.input_process = InputProcess(self.input_feats, self.latent_dim)
        self.diffusion_step_embed_dim = diffusion_step_embed_dim

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim ,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                    num_layers=self.num_layers)
  
        self.sequence_pos_encoder = PositionalEncoding(d_model=self.latent_dim , 
                                                        dropout=self.dropout,
                                                        max_len=max_len)
        self.embed_timestep = TimestepEmbedder(self.latent_dim , self.sequence_pos_encoder)
        self.output_process = OutputProcess(self.input_feats, self.latent_dim)
        self.bps_encoder = MLP_Encoder(bps_input_dim, latent_dim=latent_dim)  
        self.null_contact_token = torch.nn.Parameter(torch.randn(self.pred_horizon, latent_dim))
        self.contact_encoder = MLP_Encoder(contact_input_dim, latent_dim=latent_dim)


    def forward(self, 
                sample, 
                timestep, 
                obj_feat=None,
                contact_cond=None,
                contact_on_prob=0,
                global_states=None):
        """
        x: 
            [batch_size,  max_frames, input_feats], denoted x_t in the paper # [1, 32, 1200]
             MDM assumes x is [batch_size, njoints, nfeats, max_frames]; hence we should transpose
        timesteps: [batch_size] (int)
        """
        timesteps = timestep.expand(sample.shape[0])
        emb = self.embed_timestep(timesteps)  # diff timestep encoding [B, 1, D]
        obj_feat = torch.cat([obj_feat, global_states], axis=-1)
        obj_feat = self.bps_encoder(obj_feat) # B, T, D
        obj_feat = obj_feat.permute(1, 0, 2) # T, B, D
        if random.uniform(0, 1) < contact_on_prob: # not including 1, but include 0
            latent_contact = self.contact_encoder(contact_cond)
        else: 
            latent_contact = self.null_contact_token[None, :].repeat(emb.shape[0], 1, 1)
        latent_contact = latent_contact.permute(1, 0, 2) # T, B, D

        sample = sample.permute(1, 0, 2) # B, T, D -> T, B, D
        x = self.input_process(sample) # linear layer # [T, B, D]
        emb = emb.permute(1, 0, 2)
        x = x + obj_feat 
        x = x + latent_contact 
        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [T+1,B,D]
        xseq = self.sequence_pos_encoder(xseq)
        output = self.seqTransEncoder(xseq)[1:] 
        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output # [B, T, D]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        '''
        max_length: temporal frames; currently we only have 32 due to downsampling
        '''
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [T, B, D]
        '''
        x = x + self.pe[:x.shape[0], :] # addition
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps])


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()

        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.kpEmbedding = nn.Linear(self.input_feats, self.latent_dim)


    def forward(self, x):
        '''x: [T, B, D]
        '''
        x = self.kpEmbedding(x)
        return x

class OutputProcess(nn.Module):
    def __init__(self,input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.kpFinal = nn.Linear(self.latent_dim, self.input_feats)


    def forward(self, output):
        output = self.kpFinal(output) # T, B, D
        output = output.permute(1,0,2)  # [B, T, D]
        return output

