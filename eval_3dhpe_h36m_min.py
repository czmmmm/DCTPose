from utils.parser_util import evaluation_parser
from utils.fixseed import fixseed
from datetime import datetime
# from data_loaders.humanml.motion_loaders.model_motion_loaders import get_mdm_loader  # get_motion_loader
# from data_loaders.humanml.utils.metrics import *
# from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
# from collections import OrderedDict
# from data_loaders.humanml.scripts.motion_process import *
# from data_loaders.humanml.utils.utils import *
from utils.model_util1 import create_model_and_diffusion, load_model_wo_clip

from diffusion import logger
from utils import dist_util
from data_loaders.get_data import get_dataset_loader1
# from model.cfg_sampler import ClassifierFreeSampleModel

# torch.multiprocessing.set_sharing_strategy('file_system')
from common.utils import *
from einops import rearrange, repeat, reduce
from tqdm import tqdm
import time

def evaluate(args, model, diffusion, val_data, actions, device):
    start_eval = time.time()
    action_error_sum = define_error_list(actions)
    for i, data in enumerate(tqdm(val_data)):
        if i == 1:
            break
        batch_cam, gt_3d, cond, action, subject, scale, bb_box, cam_ind = data

        # motion = motion.to(self.device)
        gt_3d = gt_3d.to(device)
        shape = list(gt_3d.shape)
        shape[0] *= args.n_sample
        fun_sample = diffusion.ddim_sample_loop if args.sample_mode == 'ddim' else diffusion.p_sample_loop

        if args.test_augmentation:
            input_2d, output_3d = input_augmentation(cond, shape)
        else:
            cond = {'y': cond.to(device)}
            if args.n_sample > 1:
                cond['y'] = repeat(cond['y'], 'b f j d -> (n b) f j d', n=args.n_sample)
            output_3d = fun_sample(model, shape, model_kwargs=cond)
            # output_3d = diffusion.p_sample_loop(model, shape, model_kwargs=cond)

        # if args.n_sample > 1:
            # output_3d = rearrange(output_3d, '(n b) f j d -> n b f j d', n=args.n_sample)
            # output_3d = reduce(output_3d, 'n b f j d -> b f j d', 'mean')

        out_target = gt_3d.clone()
        out_target[:, :, 0] = 0

        # 之前未将预测3D根节点置零
        output_3d[:, :, 0, :] = 0

        # if self.seq2frame:
        #     out_target=out_target[:, self.nframes // 2, ...]
        #     output_3d=output_3d[:, self.nframes // 2, ...]

        if args.n_sample > 1:
            out_target = repeat(out_target, 'b f j d -> (n b) f j d', n=args.n_sample)
            # output_3d = rearrange(output_3d, '(n b) f j d -> n b f j d', n=args.n_sample)
            # for i in range(args.n_sample):
                # temp

        action_error_sum = test_calculation_eval(output_3d, out_target, action, action_error_sum, args.dataset, subject, MAE=True, n_sample=args.n_sample)

    p1, p2 = print_error(args.dataset, action_error_sum, is_train=True)

    end_eval = time.time()
    print(f'Evaluation time: {round(end_eval-start_eval)/60}min')

    return p1, p2

def input_augmentation(self, input_2D, shape):
    batch_size = input_2D.shape[0]
    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]

    input_2D_non_flip = input_2D[:, 0]
    input_2D_flip = input_2D[:, 1]
    input_2D_non_flip = {'y':input_2D_non_flip.to(self.device)}
    input_2D_flip = {'y':input_2D_flip.to(self.device)}

    output_3D_non_flip = self.diffusion.p_sample_loop(self.model, shape, model_kwargs=input_2D_non_flip)
    output_3D_flip = self.diffusion.p_sample_loop(self.model, shape, model_kwargs=input_2D_flip)

    output_3D_flip[:, :, :, 0] *= -1
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]

    output_3D = (output_3D_non_flip + output_3D_flip) / 2

    input_2D = input_2D_non_flip

    return input_2D, output_3D


if __name__ == '__main__':
    args = evaluation_parser()
    fixseed(args.seed)
    # args.batch_size = 8 # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    log_file = os.path.join(os.path.dirname(args.model_path), 'eval_humanml_{}_{}'.format(name, niter))
    # if args.guidance_param != 1.:
    #     log_file += f'_gscale{args.guidance_param}'
    # log_file += f'_{args.eval_mode}'
    log_file += '.log'

    print(f'Will save to log file [{log_file}]')

    dist_util.setup_dist(args.device)
    logger.configure()

    logger.log("creating data loader...")
    split = 'test'
    args.train = 0
    actions = define_actions(args.actions)

    val_data = get_dataset_loader1(args)

    # gt_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, hml_mode='gt')
    # gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, hml_mode='eval')
    # num_actions = gen_loader.dataset.num_actions

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args)

    logger.log(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    # if args.guidance_param != 1:
    #     model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    # eval_motion_loaders = {
    #     ################
    #     ## HumanML3D Dataset##
    #     ################
    #     'vald': lambda: get_mdm_loader(
    #         model, diffusion, args.batch_size,
    #         gen_loader, mm_num_samples, mm_num_repeats, gt_loader.dataset.opt.max_motion_length, num_samples_limit, args.guidance_param
    #     )
    # }

    # eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
    # evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times, diversity_times, mm_num_times, run_mm=run_mm)
    p1, p2 = evaluate(args, model, diffusion, val_data, actions, dist_util.dev())
    print('p1: %.2f, p2: %.2f' % (p1, p2))

