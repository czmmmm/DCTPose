from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate

def get_dataset_class(name):
    if name == "amass":
        from .amass import AMASS
        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train'):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        return t2m_collate
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train'):
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train'):
    dataset = get_dataset(name, num_frames, split, hml_mode)
    collate = get_collate_fn(name, hml_mode)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True, collate_fn=collate
    )

    return loader

import torch
from common.load_data_hm36 import Fusion
from common.h36m_dataset import Human36mDataset
def get_dataset_loader1(opt):
    actions = "*"
    root_path = opt.root_path
    dataset_path = root_path + 'data_3d_' + opt.dataset + '.npz'

    dataset = Human36mDataset(dataset_path, opt)
    actions = define_actions(actions)

    if opt.train:
        train_data = Fusion(opt=opt, train=True, dataset=dataset, root_path=root_path)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
                                                    shuffle=True, num_workers=8, pin_memory=True)

    test_data = Fusion(opt=opt, train=False, dataset=dataset, root_path=root_path)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
                                                shuffle=False, num_workers=8, pin_memory=True)
    if opt.train and opt.test:
        return train_dataloader,test_dataloader
    else:
        return test_dataloader
    
                                        
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