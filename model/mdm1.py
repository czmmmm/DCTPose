import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
# import clip
import time
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# from model.rotation2xyz import Rotation2xyz
from model.block.vanilla_transformer_encoder import Transformer
from timm.models.vision_transformer import Attention, Mlp
from common.utils import get_dct_matrix



class MDM(nn.Module):
    def __init__(self, args, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_mult=2, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):
        super().__init__()

        self.args = args
        self.legacy = legacy
        self.modeltype = modeltype
        self.n_sample = kargs.get('n_sample')
        self.njoints = njoints
        # self.njoints = njoints * self.n_sample
        self.nframes = kargs.get('nframes', None)
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.mlp_ratio = ff_mult
        self.ff_size = ff_mult*latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats_base = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.kp2d_emb = kargs.get('kp2d_emb', None)
        self.cond_merge = kargs.get('cond_merge', None)

        if self.cond_merge in ['merge1', 'merge2']:
            self.input_feats = self.input_feats_base + self.njoints * 2
        else:
            self.input_feats = self.input_feats_base

        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0

        self.in_dim = self.latent_dim // 2 if self.cond_merge == 'merge_ca' else self.latent_dim
        self.cond_dim = self.latent_dim // 2 if self.cond_merge == 'merge_ca' else self.latent_dim

        if self.arch in ['mlp_s', 'mixste_1', 'st_trans_1'] :
            self.input_process = InputProcess(self.data_rep, 3, self.in_dim)
        elif self.arch in ['mlp_mixer']:
            self.input_process = InputProcess(self.data_rep, self.input_feats, self.in_dim, fc_type='conv1d')
        elif self.arch == 'trans_st':
            self.input_process = Rearrange('b f j c -> (j c) b f')
        else:
            self.input_process = InputProcess(self.data_rep, self.input_feats, self.in_dim)

        if self.arch in ['st_trans_1']:
            self.sequence_pos_encoder = PositionalEncoding(self.njoints*self.latent_dim, self.dropout)
        # elif self.arch in ['trans_st']:
        #     self.sequence_pos_encoder = PositionalEncoding(self.nframes, self.dropout)
        else:
            self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec

        # if self.

        if self.arch == 'mixste_1':
            print('MixSTE init')
            self.spatial_pos_embed = nn.Parameter(torch.zeros(1, 2*self.njoints, self.latent_dim))
            self.temporal_pos_embed = nn.Parameter(torch.zeros(1, self.nframes, self.latent_dim))
            spatial_encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                               nhead=self.num_heads,
                                                               dim_feedforward=self.ff_size,
                                                               dropout=self.dropout,
                                                               activation=self.activation)
            temporal_encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)
            self.st_encoder = nn.ModuleList([
                nn.ModuleList([spatial_encoder_layer,
                temporal_encoder_layer])
            for _ in range(self.num_layers)])
            self.spatial_norm = nn.LayerNorm(self.latent_dim)
            self.temporal_norm = nn.LayerNorm(self.latent_dim)

        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')


    def initialize_weights(self):
        pass


    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def encode_kp2d(self, kp_2d):
        # kp_2d = rearrange(kp_2d, 'b f n d -> b f (n d)')
        b, f, n, d = kp_2d.shape

        if self.kp2d_emb == 'linear':
            if self.arch == 'mlp_s' or self.arch == 'mixste_1':
                kp_2d = rearrange(kp_2d, 'b f n d -> (b f) n d')
                return self.embed_kp2d(kp_2d).permute(1, 0, 2)
            else:
                kp_2d = rearrange(kp_2d, 'b f n d -> b f (n d)')
                return  self.embed_kp2d(kp_2d).permute(1, 0, 2)
        elif self.kp2d_emb == 'conv1d':
            kp_2d = rearrange(kp_2d, 'b f n d -> b (n d) f')
            return self.embed_kp2d(kp_2d).permute(2, 0, 1)

        elif self.kp2d_emb == 'trans_enc':
            # x = rearrange(kp_2d, 'b f d -> f b d')
            cond_x = self.cond_input_process(kp_2d)
            cond_x = self.cond_pos_encoder(cond_x)
            return self.condTrasnEncoder(cond_x)
        elif self.kp2d_emb == 'mlp':
            kp_2d = rearrange(kp_2d, 'b f n d -> b (n d) f')
            return self.embed_kp2d(kp_2d).permute(2, 0, 1)
        elif self.kp2d_emb == 'mlp_mixer':
            kp_2d = rearrange(kp_2d, 'b f n d -> b (n d) f')
            kp_2d = self.kp2d_int(kp_2d)
            p_2d = rearrange(kp_2d, 'b d f -> b f d')

            if self.arch == 'trans_base_fuse_st0':
                temp_token = repeat(self.temp_token, '1 1 d -> b 1 d', b=b)
                kp_2d = torch.cat((temp_token, kp_2d), dim=1)

            result = []
            for mb in self.embed_kp2d:
                kp_2d = mb(kp_2d)
                result.append(kp_2d.permute(1, 0, 2))
            # self.LN_cond(kp_2d)
            return kp_2d.permute(1, 0, 2), result
        elif self.kp2d_emb == 'mae':
            x = rearrange(kp_2d, 'b f n d -> b (n d) f')
            x = self.encoder(x)
            x = x.permute(0, 2, 1).contiguous()
            x = self.Transformer(x)
            return x.permute(1, 0, 2)

    def dct_process(self, x):
        b, f, n, d = x.shape
        x = rearrange(x, 'b f n d -> b f (n d)')
        dct_m, idct_m = get_dct_matrix(self.nframes)
        dct_m = dct_m.float().to(self.args.device)
        idct_m = idct_m.float().to(self.args.device)
        x = torch.matmul(dct_m[:self.args.dct_n], x)

        x = rearrange(x, 'b f d -> b (f d)')
        x = self.dct_embed(x)
        x = repeat(x, 'b d -> b n d', n=self.args.dct_n)

        output = torch.matmul(idct_m[:, :self.args.dct_n], x)

        return output

    def mixste_forward(self, x):

        s_blk = self.st_encoder[0][0]
        t_blk = self.st_encoder[0][1]

        # start_loop0 = time.time()
        x += self.spatial_pos_embed
        x = s_blk(x)
        x = self.spatial_norm(x)
        x = rearrange(x, '(b f) n d -> (b n) f d', f=self.nframes)
        # end_s0 = time.time()

        x += self.temporal_pos_embed
        x = t_blk(x)
        x = self.temporal_norm(x)
        x = rearrange(x, '(b n) f d -> (b f) n d', n=2*self.njoints)
        # end_loop0 = time.time()

        # start_loopall = time.time()
        for i in range(1, self.num_layers):
            s_blk = self.st_encoder[i][0]
            t_blk = self.st_encoder[i][1]

            x = s_blk(x)
            x = self.spatial_norm(x)
            x = rearrange(x, '(b f) n d -> (b n) f d', f=self.nframes)

            x = t_blk(x)
            x = self.temporal_norm(x)
            x = rearrange(x, '(b n) f d -> (b f) n d', n=2*self.njoints)
        # end_loopall = time.time()

        # print("spatial 0: {}ms".format((end_s0-start_loop0)*1000))
        # print("loop 0: {}ms".format((end_loop0-start_loop0)*1000))
        # print("loop last all: {}ms".format((end_loopall - start_loopall)*1000))

        return x


    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        # bs, njoints, nfeats, nframes = x.shape
        bs, nframes, njoints, nfeats = x.shape

        # start_ts = time.time()
        if self.arch == 'mlp_s' or self.arch == 'mixste_1':
            timesteps = repeat(timesteps, 'b -> b f', f=self.nframes)
            timesteps = rearrange(timesteps, 'b f -> (b f)')
        emb_t = self.embed_timestep(timesteps)  # [1, bs, d]
        # end_ts = time.time()
        # print("embed timesteps:{}ms".format((end_ts-start_ts)*1000))

        # start_cond = time.time()
        if 'kp2d' in self.cond_mode and self.kp2d_emb != '':
            if self.kp2d_emb == 'mlp_mixer':
                enc_kp2d, kp2d_result = self.encode_kp2d(y)
            else:
                enc_kp2d = self.encode_kp2d(y)
            if self.cond_merge != 'merge_ca':
                emb = emb_t + enc_kp2d

        # if self.kp2d_emb == 'mlp_mixer':
        #     enc_kp2d, kp2d_result = self.encode_kp2d(y)
        # else:
        #     enc_kp2d = self.encode_kp2d(y)
        # if self.cond_merge != 'merge_ca':
        #     emb = emb + enc_kp2d
        # end_cond = time.time()
        # print("cond encoder:{}ms".format((end_cond-start_cond)*1000))

        if self.cond_merge in ['merge1', 'merge2']:
            x = torch.cat((x, y), axis=-1)
            emb = emb_t

        # start_x = time.time()
        x = self.input_process(x)
        # end_x = time.time()
        # print("inputprocess: {}ms".format((end_x - start_x)*1000))

        if self.arch == 'dct_pure':
            pass

        elif self.arch == 'trans_enc':
            # adding the timestep embed
            # start_trans = time.time()
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            # output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
            output = self.seqTransEncoder(xseq)[nframes:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
            # for blk in self.seqTransEncoder:
            #     xseq = blk(xseq)
            # output = xseq[nframes:]

            # end_trans = time.time()
            # print("trans_enc: {}ms".format((end_trans - start_trans)*1000))
        elif self.arch == 'trans_saca0':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            emb = emb_t.permute(1, 0, 2).contiguous()
            cond = enc_kp2d.permute(1, 0, 2)
            for block in self.transblocks:
                x = block(x, cond, emb)
            output = rearrange(x, 'b f c -> f b c')

        elif self.arch == 'trans_saca1':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            emb = emb.permute(1, 0, 2).contiguous()
            cond = enc_kp2d.permute(1, 0, 2)
            for block in self.transblocks:
                x = block(x, cond, emb)
            output = rearrange(x, 'b f c -> f b c')

        elif self.arch == 'trans_saca2':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            emb = emb.permute(1, 0, 2).contiguous()
            cond = enc_kp2d.permute(1, 0, 2)
            x = x + cond
            for block in self.transblocks:
                x = block(x, cond, emb)
            output = rearrange(x, 'b f c -> f b c')

        elif self.arch == 'trans_saca3':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            emb = emb.permute(1, 0, 2).contiguous()
            cond = enc_kp2d.permute(1, 0, 2)
            x = x + emb
            for block in self.transblocks:
                x = block(x, cond, emb)
            output = rearrange(x, 'b f c -> f b c')

        elif self.arch == 'trans_saca3_skip':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            emb = emb.permute(1, 0, 2).contiguous()
            cond = enc_kp2d.permute(1, 0, 2)
            x = x + emb
            prelist = []
            for idx, block in enumerate(self.transblocks):
                if idx < (self.num_layers // 2):
                    prelist.append(x)
                    x = block(x, cond, emb)
                elif idx >= (self.num_layers // 2):
                    x = block(x, cond, emb)
                    x += prelist[-1]
                    prelist.pop()
            output = rearrange(x, 'b f c -> f b c')

        elif self.arch == 'trans_saca3_dct':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            emb = emb.permute(1, 0, 2).contiguous()
            cond = enc_kp2d.permute(1, 0, 2)
            cond_dct = self.dct_process(y)
            cond = cond + cond_dct
            x = x + emb
            for block in self.transblocks:
                x = block(x, cond, emb)
            output = rearrange(x, 'b f c -> f b c')

        elif self.arch == 'trans_base_fuse_dit_sca3':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            emb = emb.permute(1, 0, 2).contiguous()
            cond = enc_kp2d.permute(1, 0, 2)
            x = x + emb
            for idx, block in enumerate(self.transblocks):
                x = block(x)
                if idx == 3:
                    x = self.fuseblock(x, cond, emb)
            output = rearrange(x, 'b f c -> f b c')

        elif self.arch == "mlp_mixer_ada1":
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            emb = emb.permute(1, 0, 2).contiguous()
            # cond = enc_kp2d.permute(1, 0, 2)
            x = x + emb
            for block, mod in zip(self.MlpMixerEncoder, self.condmods):
                x = block(x)
                x = mod(x, emb)
            output = rearrange(x, 'b f c -> f b c')


        elif self.arch == 'trans_ada1':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            emb = emb.permute(1, 0, 2).contiguous()
            cond = enc_kp2d.permute(1, 0, 2)
            for idx, block in enumerate(self.transblocks):
                x = block(x, cond, emb)
            output = rearrange(x, 'b f c -> f b c')

        elif self.arch == 'trans_ada2':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            emb = emb.permute(1, 0, 2).contiguous()
            cond = enc_kp2d.permute(1, 0, 2)
            x = x + emb
            for idx, block in enumerate(self.transblocks):
                x = block(x, cond, emb)
            output = rearrange(x, 'b f c -> f b c')

        elif self.arch == 'trans_ada3':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            emb = emb.permute(1, 0, 2).contiguous()
            cond = enc_kp2d.permute(1, 0, 2)
            for idx, block in enumerate(self.transblocks):
                x = block(x, cond, emb)
            output = rearrange(x, 'b f c -> f b c')

        elif self.arch == 'trans_ada4':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            emb = emb.permute(1, 0, 2).contiguous()
            cond = enc_kp2d.permute(1, 0, 2)
            for idx, block in enumerate(self.transblocks):
                x = block(x, cond, emb)
            output = rearrange(x, 'b f c -> f b c')

        elif self.arch == 'trans_saca4':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            emb = emb.permute(1, 0, 2).contiguous()
            cond = enc_kp2d.permute(1, 0, 2)
            x = x + emb
            for block in self.transblocks:
                x = block(x, emb, emb)
            output = rearrange(x, 'b f c -> f b c')

        elif self.arch == 'trans_base':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            c = emb.permute(1, 0, 2).contiguous()
            x = x + c
            x = self.seqTransEncoder(x)
            output = rearrange(x, 'b f c -> f b c')

        elif self.arch == 'trans_base_verify':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            c = emb.permute(1, 0, 2).contiguous()
            x = x + c
            for idx, block in enumerate(self.transblocks):
                x = block(x)
            output = rearrange(x, 'b f c -> f b c')

        elif self.arch == 'trans_base_fuse_st0':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            c = emb.permute(1, 0, 2).contiguous()
            # TODO temp: c[:,0,:],
            x = x + c
            for idx, block in enumerate(self.transblocks):
                x = block(x)
            output = rearrange(x, 'b f c -> f b c')

        elif self.arch == 'trans_fuse_cat0':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            c = emb.permute(1, 0, 2).contiguous()
            x = x + c
            for idx, block in enumerate(self.transblocks):
                x = block(x)
                if idx < 7:
                    x = self.concat(c, x)
            output = rearrange(x, 'b f c -> f b c')

        elif self.arch == 'trans_fuse_cat1':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            c = emb.permute(1, 0, 2).contiguous()
            # x = x + c
            x = self.concat(c, x)
            for idx, block in enumerate(self.transblocks):
                x = block(x)
                if idx < 7:
                    x = self.concat(c, x)
            output = rearrange(x, 'b f c -> f b c')


        elif self.arch == 'trans_base_fuse0':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            c = emb.permute(1, 0, 2).contiguous()
            x = x + c
            for idx, block in enumerate(self.transblocks):
                x = block(x)
                if idx == 3:
                    x = self.fuseblock(x, c)
            output = rearrange(x, 'b f c -> f b c')

        elif self.arch == 'trans_base_fuse1':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            c = emb.permute(1, 0, 2).contiguous()
            x = x + c
            for idx, block in enumerate(self.transblocks):
                x = block(x)
                if idx == 3:
                    x_c = self.fuseblock1(x, c)
                    # x = self.fuseblock2(c, x)
                    x = self.fuseblock2(x_c, x)
            output = rearrange(x, 'b f c -> f b c')

        elif self.arch == 'trans_base_fuse2':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            c = emb.permute(1, 0, 2).contiguous()
            x = x + c
            for idx, block in enumerate(self.transblocks):
                x = block(x)
                if idx == 3:
                    x = x + c
            output = rearrange(x, 'b f c -> f b c')

        elif self.arch == 'trans_base_fuse_dit':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            c = emb.permute(1, 0, 2).contiguous()
            x = x + c
            for idx, block in enumerate(self.transblocks):
                # x = x = block(x)
                x = block(x)
                if idx == 3:
                    x = self.fuseblock(x, c)
            output = rearrange(x, 'b f c -> f b c')

        elif self.arch == 'trans_base_fuse_dit1':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            c = emb.permute(1, 0, 2).contiguous()
            x = x + c
            for idx, block in enumerate(self.transblocks):
                x = block(x)
                x = self.fuseblock(x, c)
                # if idx == 3:
                    # x = self.fuseblock(x, c)
            output = rearrange(x, 'b f c -> f b c')

        elif self.arch == 'trans_base_fuse_dit2':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            c = emb.permute(1, 0, 2).contiguous()
            x = x + c
            for idx, block in enumerate(self.transblocks):
                x = block(x)
                if idx <= 6:
                    x = self.fuseblock(x, c)
            output = rearrange(x, 'b f c -> f b c')

        elif self.arch == 'trans_cross0':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            c = emb.permute(1, 0, 2).contiguous()
            x = x + c
            for idx, block in enumerate(self.transblocks):
                x = block(x)
                if idx <= 6:
                    x = self.crossblock(x, c)
            output = rearrange(x, 'b f c -> f b c')
        elif self.arch == 'trans_cross1':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            c = emb.permute(1, 0, 2).contiguous()
            x = x + c
            for idx, block in enumerate(self.transblocks):
                x = block(x)
                if idx <= 6:
                    x_c = self.crossblock(x, c)
                    x = x + x_c
            output = rearrange(x, 'b f c -> f b c')

        elif self.arch == 'trans_cross2':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            c = emb.permute(1, 0, 2).contiguous()
            x = x + c
            for idx, block in enumerate(self.transblocks):
                x = block(x)
                if idx <= 6:
                    x = self.crossblock(x, c)
            output = rearrange(x, 'b f c -> f b c')

        elif self.arch == 'trans_cross3':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            c = emb.permute(1, 0, 2).contiguous()
            x = x + c
            for idx, block in enumerate(self.transblocks):
                x = block(x)
                if idx > 2 and idx <= 6:
                    x_c = self.crossblock(x, c)
                    x = x + x_c
            output = rearrange(x, 'b f c -> f b c')

        elif self.arch == 'trans_fuse':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            c = emb.permute(1, 0, 2).contiguous()

            x = self.seqTransEncoder1(x)
            fuse_feat = self.fuseblock(x, c)
            output = self.seqTransEncoder2(fuse_feat)
            output = rearrange(output, 'b f c -> f b c')

        elif self.arch == 'trans_fuse1':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            c = emb.permute(1, 0, 2).contiguous()

            for idx, block in enumerate(self.transblocks):
                x = block(x)
                if idx == 4:
                    x = self.fuseblock(x, c)
            output = rearrange(x, 'b f c -> f b c')

        elif self.arch == 'trans_fuse0':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            c = emb.permute(1, 0, 2).contiguous()
            x = self.fuseblock(x, c)
            x = self.seqTransEncoder(x)
            output = rearrange(x, 'b f c -> f b c')

        elif self.arch == 'trans_fuse2':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            c = emb.permute(1, 0, 2).contiguous()

            for idx, block in enumerate(self.transblocks):
                x = self.fuseblock(x, c)
                x = block(x)
                # if idx < self.num_layers - 1:
                    # x = self.fuseblock(x, c)
                    # x = block(x)
                # else:
                    # x = block(x)
            output = rearrange(x, 'b f c -> f b c')



        elif self.arch == 'trans_dit':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            c = emb.permute(1, 0, 2).contiguous()
            for block in self.transblocks:
                x = block(x, c)
            output = rearrange(x, 'b f c -> f b c')
        elif self.arch == 'trans_dit0':
            x = self.sequence_pos_encoder(x).permute(1, 0, 2).contiguous()
            c = emb.permute(1, 0, 2).contiguous()
            for block in self.transblocks:
                x = block(x, c)
            x = self.final_layer(x, c)
            output = rearrange(x, 'b f c -> f b c')
        elif self.arch == 'trans_st':
            # xseq = emb + x
            xseq = self.sTransEncoder(x)
            xseq = rearrange(xseq, 'd b f -> b d f')
            xseq = self.embedding(xseq).permute(2, 0, 1).contiguous()
            xseq = emb + xseq
            output = self.tTransEncoder(xseq)
        elif self.arch == 'st_trans_1':
            x += self.spatial_pos_embed.permute(1, 0, 2)
            x = self.spatial_trans_encoder(x)
            x = self.spatial_norm(x)
            x = rearrange(x, 'n (b f) c -> f b (n c)', b=bs)

            x = torch.cat((emb, x), axis=0)

            x += self.temporal_pos_embed.permute(1, 0, 2)
            x = self.temporal_trans_encoder(x)
            x = self.temporal_norm(x)
            output = x[nframes:]
        elif self.arch == 'st_trans_2':
            x = torch.cat((emb, x), axis=0) #[2*seqlen, bs, d]
            x = rearrange(x, 'f b d -> b d f')
            x = self.spatial_mlp_encoder(x)
            x = rearrange(x, 'b d f -> f b d')
            x += self.temporal_pos_embed
            x = self.temporal_trans_encoder(x)
            x = self.temporal_norm(x)
            output = x[nframes:]
            output = rearrange(output, 'f b d -> b d f')
            output = self.fcn_output(output)
            output = rearrange(output, 'b (n d) f -> b f n d', d=3)
            return output

        elif self.arch == 'mlp_mixer' or self.arch == 'mlp_mixer_1':
            if self.cond_merge == 'merge2':
                # x = torch.cat((emb, x), axis=0)
                x = emb +x
                x = rearrange(x, 'f b d -> b f d')
                for mb in self.MlpMixerEncoder:
                    x = mb(x)
                output = self.LN(x)
                output = rearrange(output, 'b f d -> f b d')
                # output = output[nframes:]
            elif self.cond_merge == 'merge_mod1':
                cond_x = repeat(enc_kp2d, 'f b d -> (repeat f) b d', repeat=2).permute(1, 0, 2)
                x = torch.cat((emb, x), axis=0)
                x = rearrange(x, 'f b d -> b f d')
                for mb in self.MlpMixerEncoder:
                    x = mb(x)
                    x = self.cond_mod(x, cond_x)
                output = self.LN(x)
                output = rearrange(output, 'b f d -> f b d')
                output = output[nframes:]
            elif self.cond_merge == 'merge_mod2':
                cond_x = repeat(enc_kp2d, 'f b d -> (repeat f) b d', repeat=2).permute(1, 0, 2)
                x = torch.cat((emb, x), axis=0)
                x = rearrange(x, 'f b d -> b f d')
                for mb in self.MlpMixerEncoder:
                    x_out = mb(x)
                    x = self.cond_mod(x_out, cond_x)
                output = self.LN(x_out)
                output = rearrange(output, 'b f d -> f b d')
                output = output[nframes:]
            elif self.cond_merge == 'merge_para':
                x = torch.cat((emb, x), axis=0)
                x = rearrange(x, 'f b d -> b f d')
                for idx, mb in enumerate(self.MlpMixerEncoder, 1):
                    x_out = mb(x)
                    if idx < self.num_layers:
                        cond_x = repeat(kp2d_result[idx], 'f b d -> (repeat f) b d', repeat=2).permute(1, 0, 2)
                        x = self.cond_mod(x_out, cond_x)
                output = self.LN(x_out)
                output = rearrange(output, 'b f d -> f b d')
                output = output[nframes:]
            elif self.cond_merge == 'merge_ca':
                x = torch.cat((enc_kp2d, x), dim=-1)
                x = emb + x
                x = rearrange(x, 'f b d -> b f d')
                for mb in self.MlpMixerEncoder:
                    x = mb(x)
                output = self.LN(x)
                output = rearrange(output, 'b f d -> f b d')

            else:
                x = torch.cat((emb, x), axis=0)
                x = rearrange(x, 'f b d -> b f d')
                for mb in self.MlpMixerEncoder:
                    x = mb(x)
                output = self.LN(x)
                output = rearrange(output, 'b f d -> f b d')
                output = output[nframes:]
        elif self.arch == 'mlp_mixer_add':
            if self.cond_merge == 'merge3':
                # x = emb + x
                emb = emb.permute(1, 0, 2)
                x = rearrange(x, 'f b d -> b f d')
                for mb in self.MlpMixerEncoder:
                    x = mb(emb+x)
                output = self.LN(x)
                output = rearrange(output, 'b f d -> f b d')

            else:
                x = emb + x
                x = rearrange(x, 'f b d -> b f d')
                for mb in self.MlpMixerEncoder:
                    x = mb(x)
                output = self.LN(x)
                output = rearrange(output, 'b f d -> f b d')

        elif self.arch == 'mlp_s':
            x = torch.cat((emb, x), axis=0)
            x = rearrange(x, 'n b d -> b n d')
            for mbe in self.MlpSEncoder:
                x = mbe(x)
            output = self.LN(x)
            output = output[:, self.njoints:, :]

        elif self.arch == 'mixste_1':

            # start_mixste1 = time.time()
            x = torch.cat((emb, x), axis=0)
            x = rearrange(x, 'n b d -> b n d')
            output = self.mixste_forward(x)
            output = output[:,self.njoints:, :]
            # end_mixste1 = time.time()
            # print("mixste-1: {}ms".format((end_mixste1 - start_mixste1)*1000))

        # start_output = time.time()
        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        # end_output = time.time()
        # print("output process: {}.ms".format((end_output-start_output)*1000))

        return output


class CondMod(nn.Module):
    def __init__(self, cond_dim, latent_dim):
        super().__init__()
        self.linear = nn.Linear(cond_dim, 2*latent_dim)

    def forward(self, x, cond):
        cond = self.linear(cond)
        cond = rearrange(cond, 'b f (r d) -> b f r d', r=2)
        x = x * (cond[:, :, 0] + 1.) + cond[:, :, 1]
        # x = x * (cond[:, :, 0] + 1.) + cond[:, :, 0]
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
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
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
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
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, fc_type=None):
        super().__init__()
        # self.data_rep = data_rep
        self.fc_type = fc_type
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        if fc_type == 'conv1d':
            self.poseEmbedding = nn.Conv1d(self.input_feats, self.latent_dim, 1, 1)
        else:
            self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        # if self.data_rep == 'rot_vel':
            # self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        # bs, njoints, nfeats, nframes = x.shape
        # x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        b, f, n, d = x.shape

        if self.input_feats == 3:
            x = rearrange(x, 'b f n d -> (b f) n d').permute(1, 0, 2)
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
        elif self.fc_type == 'conv1d':
            x = rearrange(x, 'b f n d -> b (n d) f')
            x = self.poseEmbedding(x)  # [bs, d, seqlen]
            x = rearrange(x, 'b d f -> f b d') # [seqlen, bs, d]
        else:
            x = rearrange(x, 'b f n d -> f b (n d)')
            x = self.poseEmbedding(x)  # [seqlen, bs, d]

        # if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
        # x = self.poseEmbedding(x)  # [seqlen, bs, d]

        return x
        # elif self.data_rep == 'rot_vel':
        #     first_pose = x[[0]]  # [1, bs, 150]
        #     first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
        #     vel = x[1:]  # [seqlen-1, bs, 150]
        #     vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
        #     return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        # else:
        #     raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats, nframes):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.nframes = nframes
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        # nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        # output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        # output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        if self.input_feats == 3:
            output = rearrange(output, '(b f) n d -> b f n d', f=self.nframes)
        else:
            output = rearrange(output, 'f b (n d) -> b f n d', n=self.njoints)
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output


class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.25):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        #self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w1 = nn.Conv1d(self.l_size, self.l_size, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        #self.w2 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Conv1d(self.l_size, self.l_size, kernel_size=1)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out



class SELayer(nn.Module):
    def __init__(self, c, r=4, use_max_pooling=False):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1) if not use_max_pooling else nn.AdaptiveMaxPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        bs, s, h = x.shape
        y = self.squeeze(x).view(bs, s)
        y = self.excitation(y).view(bs, s, 1)
        return x * y.expand_as(x)





class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.):
        super().__init__()
        dim_head = hidden_size // num_heads

        self.scale = dim_head ** -0.5
        self.heads = num_heads

        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x, c):
        h = self.heads

        q = self.to_q(x)

        k = self.to_k(c)
        v = self.to_v(c)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        del q, k

        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=2, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        # dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        )
        # ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class CrossBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.):
        super().__init__()

        self.attn = CrossAttention(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
        self.ff = FeedForward(dim=hidden_size, dim_out=hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, c):
        x = self.attn(self.norm1(x), c) + x
        x = self.ff(self.norm2(x)) + x

        return x

class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx) -> None:
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate  =  torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)

        ret = self._layer(x) * gate + bias
        return ret

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class StylizationBlock(nn.Module):

    def __init__(self, latent_dim, time_embed_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        # B, 1, 2D
        emb_out = self.emb_layers(emb)
        # emb_out = self.emb_layers(emb).unsqueeze(1)
        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h

class LinearTemporalSelfAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb, src_mask=None):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(x))
        # B, T, D
        # key = (self.key(self.norm(x)) + (1 - src_mask) * -1000000)
        key = self.key(self.norm(x))
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, T, H, -1), dim=1)
        # B, T, H, HD
        value = self.value(self.norm(x)).view(B, T, H, -1)
        # value = (self.value(self.norm(x)) * src_mask).view(B, T, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y


class LinearTemporalCrossAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, xf, emb):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(x))
        # B, N, D
        key = self.key(self.text_norm(xf))
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, N, H, -1), dim=1)
        # B, N, H, HD
        value = self.value(self.text_norm(xf)).view(B, N, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y

class FFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb):
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        y = x + self.proj_out(y, emb)
        return y

class LinearTemporalDiffusionTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 seq_len=60,
                 latent_dim=32,
                 text_latent_dim=512,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.1):
        super().__init__()
        self.sa_block = LinearTemporalSelfAttention(
            seq_len, latent_dim, num_head, dropout, time_embed_dim)
        self.ca_block = LinearTemporalCrossAttention(
            seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim)
        self.ffn = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)

    def forward(self, x, xf, emb, src_mask=None):
        x = self.sa_block(x, emb, src_mask)
        x = self.ca_block(x, xf, emb)
        x = self.ffn(x, emb)
        return x

class DitLinearTemporalSelfAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb, gate_msa, src_mask=None):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(x))
        # B, T, D
        # key = (self.key(self.norm(x)) + (1 - src_mask) * -1000000)
        key = self.key(self.norm(x))
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, T, H, -1), dim=1)
        # B, T, H, HD
        value = self.value(self.norm(x)).view(B, T, H, -1)
        # value = (self.value(self.norm(x)) * src_mask).view(B, T, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        y = x + gate_msa * self.proj_out(y, emb)
        return y

