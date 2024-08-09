import argparse
import os
import math
import sys
import time
import torch

class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        self.parser.add_argument('--model', type=str, default='trans_diff_2')
        # self.parser.add_argument('--model', type=str, default='trans_diff_wocond_show')
        # self.parser.add_argument('--model', type=str, default='trans_diff_wocond')
        # self.parser.add_argument('--model', type=str, default='trans_diff_1')
        # self.parser.add_argument('--model', type=str, default='trans_diff_1_fixtimeemb')
        # self.parser.add_argument('--model', type=str, default='trans_diff_fixtimeemb')
        # self.parser.add_argument('--model', type=str, default='trans_diff')
        # self.parser.add_argument('--model', type=str, default='motion_mixer_2_glow_temporal')
        # self.parser.add_argument('--model', type=str, default='motion_mixer_2_glow_temporal_1')
        # self.parser.add_argument('--model', type=str, default='trans_base_glow_temporal_1')
        # self.parser.add_argument('--model', type=str, default='trans_base_glow_temporal')
        # self.parser.add_argument('--model', type=str, default='trans_base_glow')
        # self.parser.add_argument('--model', type=str, default='motion_mixer_2_glow')
        # self.parser.add_argument('--model', type=str, default='motion_mixer_2_flow_temporal')
        # self.parser.add_argument('--model', type=str, default='motion_mixer_2_flow_gen_2loss')
        # self.parser.add_argument('--model', type=str, default='motion_mixer_2_flow_temporal')
        # self.parser.add_argument('--model', type=str, default='motion_mixer_2_flow_gen')
        # self.parser.add_argument('--model', type=str, default='trans_base_flow_gen')
        # self.parser.add_argument('--model', type=str, default='motion_mixer_2_flow1')
        # self.parser.add_argument('--model', type=str, default='trans_base_flow1')
        # self.parser.add_argument('--model', type=str, default='trans_base_flow')
        # self.parser.add_argument('--model', type=str, default='motion_mixer_2_flow_1e5')
        # self.parser.add_argument('--model', type=str, default='motion_mixer_2_flow')
        # self.parser.add_argument('--model', type=str, default='motion_mixer_2_contrastive')
        # self.parser.add_argument('--model', type=str, default='motion_mixer_2_contrastive')
        # self.parser.add_argument('--model', type=str, default='motion_mixer_1_serial_crc2d')
        # self.parser.add_argument('--model', type=str, default='motion_mixer_1_crc2d')
        # self.parser.add_argument('--model', type=str, default='motion_mixer_2_noflatten')
        # self.parser.add_argument('--model', type=str, default='motion_mixer1_pos')
        # self.parser.add_argument('--model', type=str, default='motion_mixer_2_crc2d')
        # self.parser.add_argument('--model', type=str, default='motion_mixer_2')
        # self.parser.add_argument('--model', type=str, default='motion_mixer_mlp')
        # self.parser.add_argument('--model', type=str, default='motion_mixer1')
        # self.parser.add_argument('--model', type=str, default='motion_mixer')
        # self.parser.add_argument('--model', type=str, default='mixste_rep')
        # self.parser.add_argument('--model', type=str, default='poseformer_s2s_alter_vec_3')
        # self.parser.add_argument('--model', type=str, default='poseformer_s2s_alter_vec_2')
        # self.parser.add_argument('--model', type=str, default='poseformer_s2s_alter_vec_1')
        # self.parser.add_argument('--model', type=str, default='trans_s2s_alter_vec_pt')
        # self.parser.add_argument('--model', type=str, default='poseformer_s2s_alter_vec')
        # self.parser.add_argument('--model', type=str, default='transformer_base_vec')
        # self.parser.add_argument('--model', type=str, default='transformer_base')
        # self.parser.add_argument('--model', type=str, default='slowfastpre_1_spv_s2s_alter')
        # self.parser.add_argument('--model', type=str, default='slowfastpost_spv_s2s_alter')
        # self.parser.add_argument('--model', type=str, default='slowfastpre_spv_s2s_alter')
        # self.parser.add_argument('--model', type=str, default='slow_to_fast_spv_s2s_alter')
        # self.parser.add_argument('--model', type=str, default='slow_to_fast_s2s_alter')
        # self.parser.add_argument('--model', type=str, default='slowfast_poseformer_s2s_alter')
        # self.parser.add_argument('--model', type=str, default='shift_poseformer_s2s_alter')
        # self.parser.add_argument('--model', type=str, default='shift_poseformer_s2s_alter_jsep')
        # self.parser.add_argument('--model', type=str, default='poseformer_s2s_alter_jsep')
        # self.parser.add_argument('--model', type=str, default='poseformer_s2s_alter')
        # self.parser.add_argument('--model', type=str, default='poseformer_s2s')
        # self.parser.add_argument('--model', type=str, default='poseformer')

        self.parser.add_argument('--alpha', default=8, type=int)
        # self.parser.add_argument('--slow_hid_dim', default=128, type=int)
        self.parser.add_argument('--slow_hid_dim', default=64, type=int)
        self.parser.add_argument('--fast_hid_dim', default=32, type=int)
        # self.parser.add_argument('--fast_hid_dim', default=16, type=int)
        self.parser.add_argument('--start_inter_layer', default=5, type=int)
        self.parser.add_argument('--main_layer', default=2, type=int)

        self.parser.add_argument('--lambda_slow', default=0, type=float)

        # self.parser.add_argument('--sim_dim', default=64, type=int)
        self.parser.add_argument('--sim_dim', default=32, type=int)
        self.parser.add_argument('--dim_pos', default=16, type=int)

        # self.parser.add_argument('--hid_dim', default=16, type=int)
        # self.parser.add_argument('--hid_dim', default=32, type=int)
        # self.parser.add_argument('--hid_dim', default=48, type=int)
        # self.parser.add_argument('--hid_dim', default=60, type=int)
        self.parser.add_argument('--hid_dim', default=64, type=int)
        # self.parser.add_argument('--hid_dim', default=96, type=int)
        # self.parser.add_argument('--hid_dim', default=128, type=int)
        # self.parser.add_argument('--hid_dim', default=256, type=int)
        # self.parser.add_argument('--num_layers', default=3, type=int)
        # self.parser.add_argument('--num_layers', default=4, type=int)
        # self.parser.add_argument('--num_layers', default=5, type=int)
        self.parser.add_argument('--num_layers', default=6, type=int)
        # self.parser.add_argument('--num_layers', default=7, type=int)
        # self.parser.add_argument('--num_layers', default=8, type=int)
        # self.parser.add_argument('--num_layers', default=9, type=int)
        # self.parser.add_argument('--num_layers', default=10, type=int)
        self.parser.add_argument('--num_heads', default=8, type=int)
        self.parser.add_argument('--mlp_ratio', type=int, default=4)
        # self.parser.add_argument('--mlp_ratio', type=int, default=2)
        # self.parser.add_argument('--se_ratio', type=int, default=2)
        self.parser.add_argument('--se_ratio', type=int, default=4)
        # self.parser.add_argument('--se_ratio', type=int, default=8)

        self.parser.add_argument('--crc2d', type=int, default=0)
        # self.parser.add_argument('--crc2d', type=int, default=1)
        self.parser.add_argument('--crc2d_r', type=float, default=0.)
        # self.parser.add_argument('--crc2d_r', type=float, default=0.5)
        # self.parser.add_argument('--crc2d_r', type=float, default=1.0)
        self.parser.add_argument('--dim_crc2d', default=128, type=int)
        self.parser.add_argument('--freeze', type=int, default=0)
        # self.parser.add_argument('--stage', type=int, default=1)
        self.parser.add_argument('--stage', type=int, default=2)

        # contrastive learning
        self.parser.add_argument('--contrastive', type=int, default=0)
        self.parser.add_argument('--mlp_hidden_size', type=int, default=512)
        self.parser.add_argument('--projection_hidden_size', type=int, default=128)
        self.parser.add_argument('--momentum_update', type=float, default=0.996)
        self.parser.add_argument('--checkpoint_interval', type=int, default=5000)

        # Glow
        self.parser.add_argument('--flow_pose3d_noise_ratio', default=0.005, type=float)
        self.parser.add_argument('--num_train_samples', default=2, type=int)
        self.parser.add_argument('--num_test_samples', default=1, type=int)
        self.parser.add_argument('--flow_dim', default=150, type=int)
        # self.parser.add_argument('--flow_dim', default=51, type=int)
        self.parser.add_argument('--flow_num_layers', default=4, type=int)
        self.parser.add_argument('--flow_context_features', default=3200, type=int)
        self.parser.add_argument('--flow_layer_hidden_features', default=3200, type=int)
        # self.parser.add_argument('--flow_context_features', default=1088, type=int)
        # self.parser.add_argument('--flow_layer_hidden_features', default=1088, type=int)
        self.parser.add_argument('--flow_layer_depth', default=2, type=int)
        self.parser.add_argument('--loss_weight_nll', default=0.001, type=float)
        self.parser.add_argument('--loss_weight_kp3d_exp', default=0.0, type=float)
        self.parser.add_argument('--loss_weight_kp3d_mode', default=1.0, type=float)
        self.parser.add_argument('--loss_weight_temporal', default=0.01, type=float)
    
        # Diffusion
        self.parser.add_argument('--timesteps', default=100, type=int)
        self.parser.add_argument('--loss_type', default='l2', type=str)
        # self.parser.add_argument('--objective', default='pred_noise', type=str)
        self.parser.add_argument('--objective', default='pred_x0', type=str)
        self.parser.add_argument('--cond_dim', default=128, type=int)
        self.parser.add_argument('--cond', default=1, type=int)
        self.parser.add_argument('--show', default=1, type=int)
        self.parser.add_argument('--show_bs', default=64, type=int, help='batch size of showed poses')
        self.parser.add_argument('--geometric_loss', default=1, type=int)
        self.parser.add_argument('--lambda_pos', default=1., type=float, help="Joint position loss.")

        


        self.parser.add_argument('--dataset', type=str, default='h36m')
        # self.parser.add_argument('-k', '--keypoints', default='gt', type=str)
        self.parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str)
        self.parser.add_argument('--data_augmentation', type=bool, default=True)
        self.parser.add_argument('--reverse_augmentation', type=bool, default=False)
        self.parser.add_argument('--test_augmentation', type=bool, default=False)
        self.parser.add_argument('--crop_uv', type=int, default=0)
        self.parser.add_argument('--root_path', type=str, default='data/')
        self.parser.add_argument('-a', '--actions', default='*', type=str)
        self.parser.add_argument('--downsample', default=1, type=int)
        self.parser.add_argument('--subset', default=1, type=float)
        # self.parser.add_argument('-s', '--stride', default=1, type=int)
        # self.parser.add_argument('-s', '--stride', default=9, type=int)
        self.parser.add_argument('-s', '--stride', default=27, type=int)
        # self.parser.add_argument('-s', '--stride', default=50, type=int)
        # self.parser.add_argument('-s', '--stride', default=81, type=int)
        # self.parser.add_argument('-s', '--stride', default=151, type=int)
        # self.parser.add_argument('-s', '--stride', default=243, type=int)
        # self.parser.add_argument('--gpu', default='0, 1', type=str, help='')
        # self.parser.add_argument('--gpu', default='1', type=str, help='')
        self.parser.add_argument('--gpu', default='0', type=str, help='')
        # self.parser.add_argument('--train', action='store_true')
        self.parser.add_argument('--train', type=int, default=1)
        self.parser.add_argument('--test', type=int, default=1)
        # self.parser.add_argument('--nepoch', type=int, default=1000)
        # self.parser.add_argument('--nepoch', type=int, default=500)
        self.parser.add_argument('--nepoch', type=int, default=200)
        # self.parser.add_argument('--nepoch', type=int, default=160)
        # self.parser.add_argument('--nepoch', type=int, default=80)
        # self.parser.add_argument('--nepoch', type=int, default=40)
        # self.parser.add_argument('--nepoch', type=int, default=20)
        # self.parser.add_argument('--nepoch', type=int, default=2)
        # self.parser.add_argument('--batch_size', type=int, default=2, help='can be changed depending on your machine')
        # self.parser.add_argument('--batch_size', type=int, default=32, help='can be changed depending on your machine')
        # self.parser.add_argument('--batch_size', type=int, default=64, help='can be changed depending on your machine')
        # self.parser.add_argument('--batch_size', type=int, default=96, help='can be changed depending on your machine')
        self.parser.add_argument('--batch_size', type=int, default=128, help='can be changed depending on your machine')
        # self.parser.add_argument('--batch_size', type=int, default=512, help='can be changed depending on your machine')
        # self.parser.add_argument('--batch_size', type=int, default=1024, help='can be changed depending on your machine')
        self.parser.add_argument('--dropout', default=0.5, type=float)
        # self.parser.add_argument('--lr', type=float, default=2e-2)
        # self.parser.add_argument('--lr', type=float, default=1e-2)
        # self.parser.add_argument('--lr', type=float, default=1e-3)
        # self.parser.add_argument('--lr', type=float, default=4e-4)
        # self.parser.add_argument('--lr', type=float, default=2e-4)
        # self.parser.add_argument('--lr', type=float, default=1e-4)
        self.parser.add_argument('--lr', type=float, default=1e-5)
        # self.parser.add_argument('--lr_decay_large', type=float, default=0.5)
        self.parser.add_argument('--snapshot', default=10, type=int)
        # self.parser.add_argument('--lr_decay_large', type=float, default=0.9)
        self.parser.add_argument('--lr_decay_large', type=float, default=0.95)
        self.parser.add_argument('--large_decay_epoch', type=int, default=100)
        # self.parser.add_argument('--large_decay_epoch', type=int, default=10)
        # self.parser.add_argument('-lrd', '--lr_decay', default=0.9, type=float)
        # self.parser.add_argument('-lrd', '--lr_decay', default=0.99, type=float)
        self.parser.add_argument('-lrd', '--lr_decay', default=1.0, type=float)
        self.parser.add_argument('--weight_decay', type=float, default=1e-1)
        # self.parser.add_argument('--workers', type=int, default=0)
        # self.parser.add_argument('--workers', type=int, default=12)
        # self.parser.add_argument('--workers', type=int, default=8)
        self.parser.add_argument('--workers', type=int, default=4)
        self.parser.add_argument('--frames', type=int, default=1)
        # self.parser.add_argument('--frames', type=int, default=27)
        self.parser.add_argument('--pad', type=int, default=0)
        self.parser.add_argument('--reload', action='store_true')
        self.parser.add_argument('--checkpoint', type=str, default='')
        self.parser.add_argument('--previous_dir', type=str, default='')
        # self.parser.add_argument('--previous_dir', type=str, default='/home/noteleks/workspace/st3dhpe/checkpoint/cpn_0629_1112_17_transformer_base_l6_d128_1/model_72_5137.pth')
        # self.parser.add_argument('--previous_dir', type=str, default='/home/noteleks/workspace/st3dhpe/checkpoint/cpn_0717_2239_56_motion_mixer_2_l6_d128_f50_ser4/model_185_5113.pth')
        # self.parser.add_argument('--previous_dir', type=str, default='/home/noteleks/workspace/st3dhpe/checkpoint/gt_0717_2240_54_motion_mixer_2_l6_d128_f50_ser4/model_91_3573.pth')
        # self.parser.add_argument('--previous_dir', type=str, default='/home/noteleks/workspace/st3dhpe/runs/Jul26_22-49-21_zhm_stage1/checkpoints/model.pth')
        # self.parser.add_argument('--previous_dir', type=str, default='/home/noteleks/workspace/st3dhpe/runs/Jul24_11-30-21_zhm/checkpoints/model.pth')
        # self.parser.add_argument('--previous_dir', type=str, default='/home/noteleks/workspace/st3dhpe/runs/Jul24_14-06-37_zh/checkpoints/model.pth')
        # self.parser.add_argument('--previous_dir', type=str, default='checkpoint/pretrained')
        # self.parser.add_argument('--previous_dir', type=str, default='/home/noteleks/workspace/st3dhpe/checkpoint/pretrained/cpn_0723_0016_29_motion_mixer_1_serial_crc2d_l6_d128_f50_ser4/model_197_440.pth')
        # self.parser.add_argument('--previous_dir', type=str, default='/home/noteleks/workspace/st3dhpe/checkpoint/pretrained/cpn_0723_1502_02_motion_mixer_1_serial_crc2d_l6_d128_f50_ser4/model_229_418.pth')
        # self.parser.add_argument('--pretrained', type=int, default=1)
        self.parser.add_argument('--pretrained', type=int, default=0)
        # self.parser.add_argument('--pretrained_path', type=str, default=1)
        self.parser.add_argument('--num_joints', type=int, default=17)
        self.parser.add_argument('--out_joints', type=int, default=17)
        # self.parser.add_argument('--out_all', type=int, default=0)
        self.parser.add_argument('--out_all', type=int, default=1)
        self.parser.add_argument('--in_channels', type=int, default=2)
        self.parser.add_argument('--out_channels', type=int, default=3)
        # self.parser.add_argument('--alpha', type=float, default=1.0)
        self.parser.add_argument('-previous_best_threshold', type=float, default= math.inf)
        self.parser.add_argument('-previous_name', type=str, default='')

        self.parser.add_argument('--tcloss', type=int, default=0)
        # self.parser.add_argument('--alpha', type=float, default=0.3)

    def parse(self):
        self.init()

        # self.opt = self.parser.parse_known_args()[0]
        self.opt = self.parser.parse_args()
        # self.opt = self.parser.parse_args(args=[])
        self.opt.pad = (self.opt.frames-1) // 2
        # self.opt.receptive_field =

        self.opt.subjects_train = 'S1,S5,S6,S7,S8'
        self.opt.subjects_test = 'S9,S11'

        env = sys.executable
        if env.startswith('/home/user001/'):
            self.opt.root_path = '/home/omnisky/data/everyOne/zhmchen/data/h36m/'
        
        if self.opt.keypoints.startswith('cpn'):
            kp = 'cpn'
        elif self.opt.keypoints == 'gt':
            kp = 'gt'

        if self.opt.model != 'poseformer':
            self.opt.frames = self.opt.stride
            frames = self.opt.frames
        else:
            self.opt.frames = self.opt.frames
            frames = self.opt.frames
        if self.opt.stage == 1:
            ckpt_path = 'checkpoint/pretrained/'
        else:
            ckpt_path = 'checkpoint/'

        if self.opt.train:
            logtime = time.strftime('%m%d_%H%M_%S_')
            if self.opt.model.startswith('motion'):
                self.opt.checkpoint = ckpt_path + kp + '_' + logtime + self.opt.model +  '_l' + '%d' % (self.opt.num_layers) + \
                                      '_d' + '%d' % self.opt.hid_dim + '_f' + '%d' % (frames) + '_ser' + '%d' % (self.opt.se_ratio)
            elif self.opt.model.startswith('slow_to_fast_spv') or self.opt.model.startswith('slowfastp'):
                self.opt.checkpoint = ckpt_path + kp + '_' + logtime + self.opt.model + '_spv_%.1f' % (self.opt.lambda_slow) + '_l' + '%d' % (self.opt.num_layers) + \
                                      '_sil' + '%d' % (self.opt.start_inter_layer) + '_d' + '%d' % (self.opt.slow_hid_dim) \
                                      + '_' + '%d' % (self.opt.fast_hid_dim) + '_f' + '%d' % (frames)
            elif self.opt.model.startswith('slow'):
                self.opt.checkpoint = ckpt_path + kp + '_' + logtime + self.opt.model + '_l' + '%d' % (self.opt.num_layers) + \
                                      '_sil' + '%d' % (self.opt.start_inter_layer) + '_d' + '%d' % (self.opt.slow_hid_dim)\
                                      + '_' + '%d' % (self.opt.fast_hid_dim) + '_f' + '%d' % (frames)
            else:
                self.opt.checkpoint = ckpt_path + kp + '_' + logtime + self.opt.model + '_l' + '%d'%(self.opt.num_layers)+ \
                                  '_d' + '%d'%(self.opt.hid_dim) + '_f' +'%d'%(frames)
            # self.opt.checkpoint = 'checkpoint/' + logtime + '%d'%(self.opt.frames)
            if not os.path.exists(self.opt.checkpoint):
                os.makedirs(self.opt.checkpoint)
                print(self.opt.checkpoint)

            args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                    if not name.startswith('_'))
            file_name = os.path.join(self.opt.checkpoint, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('==> Args:\n')
                for k, v in sorted(args.items()):
                    opt_file.write('  %s: %s\n' % (str(k), str(v)))
                opt_file.write('==> Args:\n')

        return self.opt


class opts_crc2d():
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
    
    def init(self):
        self.parser.add_argument('--model', type=str, default='correction_2d_fc')

        self.parser.add_argument('--hid_dim', default=128, type=int)

        self.parser.add_argument('--dataset', type=str, default='h36m')
        self.parser.add_argument('--keypoints', default='cpn_ft_h36m_dbb', type=str)
        self.parser.add_argument('--data_augmentation', type=bool, default=False)
        self.parser.add_argument('--reverse_augmentation', type=bool, default=False)
        self.parser.add_argument('--test_augmentation', type=bool, default=False)
        self.parser.add_argument('--crop_uv', type=int, default=0)
        self.parser.add_argument('--root_path', type=str, default='data/')
        self.parser.add_argument('--actions', default='*', type=str)
        self.parser.add_argument('--downsample', default=1, type=int)
        self.parser.add_argument('--subset', default=1, type=float)
        # self.parser.add_argument('--stride', default=243, type=int)
        self.parser.add_argument('--stride', default=50, type=int)
        self.parser.add_argument('--gpu', default='1', type=str)
        self.parser.add_argument('--train', type=int, default=1)
        self.parser.add_argument('--test', type=int, default=0)
        self.parser.add_argument('--nepoch', type=int, default=300)
        self.parser.add_argument('--batch_size', type=int, default=96)
        self.parser.add_argument('--dropout', default=0.5, type=float)
        self.parser.add_argument('--lr', type=float, default=1e-4)
        self.parser.add_argument('--lr_decay_large', type=float, default=0.5)
        self.parser.add_argument('--large_decay_epoch', type=int, default=10)
        self.parser.add_argument('--lr_decay', type=float, default=0.95)
        self.parser.add_argument('--workers', type=int, default=8)
        self.parser.add_argument('--frames', type=int, default=1)
        self.parser.add_argument('--pad', type=int, default=0)
        self.parser.add_argument('--reload', action='store_true')
        self.parser.add_argument('--checkpoint', type=str, default='')
        self.parser.add_argument('--prevoious_dir', type=str, default='')
        self.parser.add_argument('--n_joints', type=int, default=17)
        self.parser.add_argument('--out_joints', type=int, default=17)
        self.parser.add_argument('--out_all', type=int, default=1)
        self.parser.add_argument('--in_channels', type=int, default=2)
        self.parser.add_argument('--out_channels', type=int, default=2)
        self.parser.add_argument('--previous_best_threshold', type=float, default=math.inf)
        self.parser.add_argument('--previous_name', type=str, default='')
    
    def parse(self):
        self.init()

        self.opt = self.parser.parse_args()
        self.opt.pad = (self.opt.frames-1) // 2

        self.opt.subjects_train = 'S1,S5,S6,S7,S8'
        self.opt.subjects_test = 'S9,S11'

        if self.opt.keypoints.startswith('cpn'):
            kp = 'cpn'
        elif self.opt.keypoints == 'gt':
            kp = 'gt'
        if self.opt.out_all:
            self.opt.frames = self.opt.stride
            frames = self.opt.frames
        else:
            frames = self.opt.frames
        
        if self.opt.train:
            logtime = time.strftime('%m%d_%H%M_%S_')
            self.opt.checkpoint = 'checkpoint/pretrained/' + kp + '_' + logtime + self.opt.model + \
                '_d' + '%d' % self.opt.hid_dim + '_f' + '%d' % frames
        
        if not os.path.exists(self.opt.checkpoint):
            os.makedirs(self.opt.checkpoint)
            print(self.opt.checkpoint)
        
        args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                if not name.startswith('_'))
        file_name = os.path.join(self.opt.checkpoint, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> Args:\n')
            for k, v in sorted(args.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))
            opt_file.write('==> Args:\n')

        return self.opt

