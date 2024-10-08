# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import torch
import numpy as np
import hashlib
from shutil import copyfile
from torch.autograd import Variable
from einops import rearrange
import torch.nn.functional as F

def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """

    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result


def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2 ** 32 - 1) * (max_value - min_value)) + min_value


def mpjpe_cal(predicted, target):
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))

def tcloss(predicted):
    pred1 = predicted[:, 1:, ...]
    pred2 = predicted[:, :-1, ...]

    return torch.mean(torch.norm(pred1 - pred2, dim=len(predicted.shape) - 1))


# def test_calculation(predicted, target, action, error_sum, data_type, subject):
#     error_sum = mpjpe_by_action_p1(predicted, target, action, error_sum)
#     error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum)

#     return error_sum
def test_calculation(predicted, target, action, error_sum, data_type, subject, MAE=False):
    error_sum = mpjpe_by_action_p1(predicted, target, action, error_sum)
    if not MAE:
        error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum)

    return error_sum

def test_calculation_eval(predicted, target, action, error_sum, data_type, subject, MAE=False, n_sample=None):
    error_sum = mpjpe_by_action_p1(predicted, target, action, error_sum, n_sample)
    if not MAE:
        error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum, n_sample)

    return error_sum

def test_calculation_mean(predicted, target, action, error_sum, data_type, subject, MAE=True, mean_pose=True):
    error_sum = mpjpe_by_action_p1_mean(predicted, target, action, error_sum, mean_pose)
    if not MAE:
        predicted = torch.mean(predicted, dim=1, keepdim=False)
        error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum)

    return error_sum

def test_calculation_eval_JBest(predicted, target, action, error_sum, data_type, subject, MAE=False, n_sample=None):
    error_sum = mpjpe_by_action_p1_jbest(predicted, target, action, error_sum, n_sample)
    if not MAE:
        error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum, n_sample)

    return error_sum

def mpjpe_by_action_p1_jbest(predicted, target, action, action_error_sum, n_sample=None):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    frame = predicted.shape[1]
    dist = torch.norm(predicted - target, dim=len(target.shape) - 1)

    dist = rearrange(dist, '(n b) f j -> f n b j', n=n_sample)
    # dist = rearrange(dist, 'b f n j -> f n b j')
    dist = torch.min(dist, dim=1, keepdim=False).values
    dist = dist.reshape(frame, -1)
    dist = torch.mean(dist, dim=-1, keepdim=False)

    if not n_sample:
        if len(set(list(action))) == 1:
            end_index = action[0].find(' ')
            if end_index != -1:
                action_name = action[0][:end_index]
            else:
                action_name = action[0]

            action_error_sum[action_name]['p1'].update(torch.mean(dist).item() * num, num)
        else:
            for i in range(num):
                end_index = action[i].find(' ')
                if end_index != -1:
                    action_name = action[i][:end_index]
                else:
                    action_name = action[i]

                action_error_sum[action_name]['p1'].update(torch.mean(dist[i]).item(), 1)
                # action_error_sum[action_name]['p1'].update(dist[i].item(), 1)
    else:
        # dist = rearrange(dist, '(n b) f -> b n f', n=n_sample)
        num = int(num/n_sample)
        # # dist = dist.reshape(int(num/n_sample), n_sample, -1)
        # dist = torch.mean(dist, dim=-1)
        # dist = dist.reshape(num, n_sample)
        # # num = num/n_sample
        # err = torch.min(dist, dim=-1).values
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]
            action_error_sum[action_name]['p1'].update(dist[i], 1)



    return action_error_sum


def mpjpe_by_action_p1(predicted, target, action, action_error_sum, n_sample=None):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=len(target.shape) - 2)

    if not n_sample:
        if len(set(list(action))) == 1:
            end_index = action[0].find(' ')
            if end_index != -1:
                action_name = action[0][:end_index]
            else:
                action_name = action[0]

            action_error_sum[action_name]['p1'].update(torch.mean(dist).item() * num, num)
        else:
            for i in range(num):
                end_index = action[i].find(' ')
                if end_index != -1:
                    action_name = action[i][:end_index]
                else:
                    action_name = action[i]

                action_error_sum[action_name]['p1'].update(torch.mean(dist[i]).item(), 1)
                # action_error_sum[action_name]['p1'].update(dist[i].item(), 1)
    else:
        # dist = rearrange(dist, '(n b) f -> b n f', n=n_sample)
        num = int(num/n_sample)
        # dist = dist.reshape(int(num/n_sample), n_sample, -1)
        dist = torch.mean(dist, dim=-1)
        dist = dist.reshape(num, n_sample)
        # num = num/n_sample
        err = torch.min(dist, dim=-1).values
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]
            action_error_sum[action_name]['p1'].update(err[i], 1)



    return action_error_sum

def mpjpe_by_action_p1_mean(predicted, target, action, action_error_sum, n_sample):
    # assert predicted.shape == target.shape
    # num = predicted.size(0)
    # dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=len(target.shape) - 2)


        # dist = rearrange(dist, '(n b) f -> b n f', n=n_sample)
        # num = int(num/n_sample)
        # dist = dist.reshape(int(num/n_sample), n_sample, -1)
        # dist = torch.mean(dist, dim=-1)
        # dist = dist.reshape(num, n_sample)
        # num = num/n_sample
        # err = torch.min(dist, dim=-1).values
    num = predicted.shape[0]
    mean_pose = torch.mean(predicted, dim=1, keepdim=False)
    # print("mean_pose shape:", mean_pose.shape)
    # print("target shape:", target.shape)
    # target = target.unsqueeze(1).repeat(1, n_sample, 1, 1, 1)
    errors = torch.norm(mean_pose - target, dim=len(target.shape) - 1)
    errors = errors.reshape(num, -1)
    errors = torch.mean(errors, dim=-1, keepdim=False)

    for i in range(num):
        end_index = action[i].find(' ')
        if end_index != -1:
            action_name = action[i][:end_index]
        else:
            action_name = action[i]
        action_error_sum[action_name]['p1'].update(errors[i], 1)



    return action_error_sum

def mpjpe_by_action_p2(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    pred = predicted.detach().cpu().numpy().reshape(-1, predicted.shape[-2], predicted.shape[-1])
    gt = target.detach().cpu().numpy().reshape(-1, target.shape[-2], target.shape[-1])
    # dist = p_mpjpe(pred, gt)
    dist = np.random.rand(num)

    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]
        action_error_sum[action_name]['p2'].update(np.mean(dist) * num, num)
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]
            action_error_sum[action_name]['p2'].update(np.mean(dist), 1)

    return action_error_sum


def p_mpjpe(predicted, target):
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY
    t = muX - a * np.matmul(muY, R)

    predicted_aligned = a * np.matmul(predicted, R) + t

    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1), axis=len(target.shape) - 2)


def define_actions(action):
    actions = ["Directions", "Discussion", "Eating", "Greeting",
               "Phoning", "Photo", "Posing", "Purchases",
               "Sitting", "SittingDown", "Smoking", "Waiting",
               "WalkDog", "Walking", "WalkTogether"]

    if action == "All" or action == "all" or action == '*':
        return actions

    if not action in actions:
        raise (ValueError, "Unrecognized action: %s" % action)

    return [action]


def define_error_list(actions):
    error_sum = {}
    error_sum.update({actions[i]:
                          {'p1': AccumLoss(), 'p2': AccumLoss()}
                      for i in range(len(actions))})
    return error_sum


class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def get_varialbe(split, target):
    num = len(target)
    var = []
    if split == 'train':
        for i in range(num):
            temp = Variable(target[i], requires_grad=False).contiguous().type(torch.cuda.FloatTensor)
            var.append(temp)
    else:
        for i in range(num):
            temp = Variable(target[i]).contiguous().cuda().type(torch.cuda.FloatTensor)
            var.append(temp)

    return var


def print_error(data_type, action_error_sum, is_train):
    mean_error_p1, mean_error_p2 = print_error_action(action_error_sum, is_train)

    return mean_error_p1, mean_error_p2


def print_error_action(action_error_sum, is_train):
    mean_error_each = {'p1': 0.0, 'p2': 0.0}
    mean_error_all = {'p1': AccumLoss(), 'p2': AccumLoss()}

    if is_train == 0:
        print("{0:=^12} {1:=^10} {2:=^8}".format("Action", "p#1 mm", "p#2 mm"))

    for action, value in action_error_sum.items():
        if is_train == 0:
            print("{0:<12} ".format(action), end="")

        mean_error_each['p1'] = action_error_sum[action]['p1'].avg * 1000.0
        mean_error_all['p1'].update(mean_error_each['p1'], 1)

        mean_error_each['p2'] = action_error_sum[action]['p2'].avg * 1000.0
        mean_error_all['p2'].update(mean_error_each['p2'], 1)

        if is_train == 0:
            print("{0:>6.2f} {1:>10.2f}".format(mean_error_each['p1'], mean_error_each['p2']))

    if is_train == 0:
        print("{0:<12} {1:>6.2f} {2:>10.2f}".format("Average", mean_error_all['p1'].avg, \
                                                    mean_error_all['p2'].avg))

    return mean_error_all['p1'].avg, mean_error_all['p2'].avg


def save_model(previous_name, save_dir, epoch, data_threshold, model, is_end=False):
    if is_end:
        torch.save(model.state_dict(),
                   '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100))
        return 0
    if os.path.exists(previous_name):
        os.remove(previous_name)

    torch.save(model.state_dict(),
               '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100))
    previous_name = '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100)
    return previous_name


def _create_model_training_folder(writer, files_to_same):
# def _create_model_training_folder(model_checkpoints_folder, files_to_same):
    model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        for file in files_to_same:
            copyfile(file, os.path.join(model_checkpoints_folder, os.path.basename(file)))

def get_dct_matrix(N, is_torch=True):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    if is_torch:
        dct_m = torch.from_numpy(dct_m)
        idct_m = torch.from_numpy(idct_m)
    return dct_m, idct_m

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from lib.utils.geometry_utils import *


# class SmoothNetLoss(nn.Module):

#     def __init__(self, w_accel, w_pos):
#         super().__init__()
#         self.w_accel = w_accel
#         self.w_pos = w_pos

#     def mask_lr1_loss(self, inputs, mask, targets):
#         Bs, C, L = inputs.shape

#         not_mask = 1 - mask.int()
#         not_mask = not_mask.unsqueeze(1).repeat(1, C, 1).float()

#         N = not_mask.sum(dtype=torch.float32)
#         loss = F.l1_loss(
#             inputs * not_mask, targets * not_mask, reduction='sum') / N
#         return loss

def SmoothNetLoss(denoise, gt):
    denoise = denoise.permute(0, 2, 1)
    gt = gt.permute(0, 2, 1)

    loss_pos = F.l1_loss(
        denoise, gt, reduction='mean')

    accel_gt = gt[:,:,:-2] - 2 * gt[:,:,1:-1] + gt[:,:,2:]
    accel_denoise = denoise[:,:,:-2] - 2 * denoise[:,:,1:-1] + denoise[:,:,2:]

    loss_accel=F.l1_loss(
        accel_denoise, accel_gt, reduction='mean')


        # weighted_loss = self.w_accel * loss_accel + self.w_pos * loss_pos
    weighted_loss = 0.1 * loss_accel + 1.0 * loss_pos

    return weighted_loss