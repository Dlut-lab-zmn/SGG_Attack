# Copyright (c) Mn,Zhao. All Rights Reserved.
from __future__ import division

from builtins import range
from concurrent.futures import thread
import logging
from collections import Iterable
from pickle import NONE
import time
import get_grad
import numpy as np
from attack_base import Attack
import torch
import get_loss
from torch.nn.parameter import Parameter
__all__ = [
    'Gradient_CrossPatch_Attack', 'GCPA',
    'Gradient_RecPatch_Attack', 'GRPA', 
    'BasicIterativeMethodAttack', 'BIM', 
    'MomentumIteratorAttack', 'MIFGSM', 
    'FastGradientSignMethodAttack', 'FGSM',
    'B_Test' ,'BasicTest'
]


class BasicIterativeMethodAttack(Attack):

    def __init__(self, model):

        super(BasicIterativeMethodAttack, self).__init__(model)
        self.initial_perturbation = False
        self.init_weight = 0
        self.use_grad_mask = False
        self.grad_mask = None
        self.dynamic_select = False
        self.dynamic_ratio = 1
        self.mask_band = 0
        self.ori_mask = False
    def __call__(self, input,*args,  **kwargs):
        return self._apply( input,*args,  **kwargs)
    def _apply(self, images,targets,
               normal,model_ls,
               attack_mode,
               norm_ord='inf',
               epsilons=0.1,
               epsilons_max=1.0,
               budget = 0.1,
               steps=20,
               epsilon_steps=5, 
               min_=0, max_ =  255,
                is_validate = False,clip_each_iter = False,int_stype = False):
        # print(self.model.name)
        if norm_ord == 0:
            raise ValueError("L0 norm is not supported!")

        logging.info('epsilons={0},epsilons_max={1},steps={2},epsilon_steps={3}'.
                     format(epsilons,epsilons_max,steps,epsilon_steps))

        if not isinstance(epsilons, Iterable):
            #从epsilons到0.5逐步增大
            epsilons = np.linspace(epsilons, epsilons_max, num=epsilon_steps)

        #从[epsilon,0.5]动态调整epsilon 直到攻击成功
        step = 0
        input = images.tensors
        adv_img  = input
        attack_record = {'attack_mode': attack_mode, 'best_adv_img': adv_img, 'ar_ratio':[0,0,0,0,0],'best_rel_ratio': [0.,0.,0.,0.,0.], 'best_obj_ratio': [0.,0.,0.,0.,0.]}
        attack_targets = targets[0]
        source_targets = targets[1]
        is_success = False
        if self.use_grad_mask:
            patch_select = Patch_select()
            sub_patch_select = getattr(patch_select,self.grad_mask)
            constant_grad_mask = sub_patch_select(adv_img, attack_targets, source_targets, mask_band = self.mask_band)
        else:
            constant_grad_mask = torch.ones_like(adv_img)
        if self.initial_perturbation:
            initial_perturbation = torch.randn_like(adv_img) * (max_ - min_) * self.init_weight
        else:
            initial_perturbation = torch.zeros_like(adv_img)
        Total_perturbation = initial_perturbation
        GT_triplets = self.get_GT_triplet(attack_targets)
        for epsilon in epsilons[:]:
            if epsilon == 0.0:
                continue
            for i in range(steps):
                # adv_img = torch.tensor(adv_img, requires_grad=True)
                adv_img = adv_img.clone().detach().requires_grad_(True)
                get_loss_result = get_loss.get_loss(self.model,adv_img,images, attack_targets, attack_mode)

                if get_loss_result is None:
                    print('No proposal pairs')
                    return is_success, attack_record, proposals,proposals_pairs
                loss, proposals ,proposals_pairs= get_loss_result
                gradient = - get_grad.get_grad(adv_img, loss, self.model)

                if norm_ord == 'inf':
                    gradient_norm = torch.sign(gradient)
                else:
                    gradient_norm = gradient / self._norm(gradient, ord=norm_ord)
                Total_perturbation += epsilon * gradient_norm
                Total_perturbation = torch.clip(Total_perturbation, -int(budget*255), int(budget*255))
                grad_mask = constant_grad_mask
                if self.dynamic_select:
                    Total_gradient_mask = torch.sum(torch.abs(Total_perturbation),1, keepdim=True) * grad_mask
                    Total_gradient_mask += torch.randn_like(Total_gradient_mask)/10000
                    Thread = torch.sort(Total_gradient_mask.view(-1), descending = True)[0][ int(self.dynamic_ratio *len(Total_gradient_mask.view(-1).nonzero())) - 1]# + 1e-5
                    grad_mask = Total_gradient_mask >= Thread
                    #print(torch.sort(Total_gradient_mask.view(-1), descending = True)[0])
                    #print(Total_gradient_mask >= Thread)
                #print(grad_mask)
                if self.ori_mask:
                    adv_img = input * (~grad_mask) + Total_perturbation * grad_mask
                else:
                    adv_img = input # + Total_perturbation * grad_mask

                if clip_each_iter:
                    adv_img = self.recover_img(adv_img, normal, min_, max_)
                        if int_stype:
                            adv_img = adv_img.int()
                    adv_img = self.preprocess_img(adv_img,normal)
                is_success = False
                updated_flag = False
                proposals = None
                proposals_pairs = None
                if is_validate:
                    evaluate_result = self.is_success(attack_record,model_ls,GT_triplets,adv_img, images, attack_targets)
                    if evaluate_result is not None:
                        is_success, updated_flag, proposals ,proposals_pairs = evaluate_result
                if updated_flag:
                    attack_record['best_adv_img'] = adv_img
                if is_success:
                    return is_success, attack_record, proposals,proposals_pairs
            step += 1
        return is_success, attack_record, proposals,proposals_pairs


class ProjectGradientDAttack(BasicIterativeMethodAttack):

    def __init__(self, model):
        super(ProjectGradientDAttack, self).__init__(model)
        self.initial_perturbation = True
        self.init_weight = 0.1

class Gradient_CrossPatch_Attack(BasicIterativeMethodAttack):

    def __init__(self, model):

        super(Gradient_CrossPatch_Attack, self).__init__(model)
        self.use_grad_mask = True
        self.grad_mask = 'cross_patch_select'
        self.mask_band = 10
class Gradient_RecPatch_Attack(BasicIterativeMethodAttack):

    def __init__(self, model):

        super(Gradient_RecPatch_Attack, self).__init__(model)
        self.use_grad_mask = True
        self.grad_mask = 'rec_patch_select'
        self.mask_band = 10

class Gradient_CrossXPatch_Attack(BasicIterativeMethodAttack):

    def __init__(self, model):

        super(Gradient_CrossXPatch_Attack, self).__init__(model)
        self.use_grad_mask = True
        self.grad_mask = 'cross_X_patch_select'
        self.mask_band = 10

class SparsityBasicIterativeMethodAttack(BasicIterativeMethodAttack):
    def __init__(self, model):
        super(SparsityBasicIterativeMethodAttack, self).__init__(model)
        self.dynamic_select = True
        self.dynamic_ratio = 0.1
class SparsityBasicIterativeMethodAttack2(BasicIterativeMethodAttack):
    def __init__(self, model):
        super(SparsityBasicIterativeMethodAttack2, self).__init__(model)
        self.dynamic_select = True
        self.ori_mask = True
        self.dynamic_ratio = 0.1
class Patch_select(object):

    def cross_patch_select(self, input, attack_targets, source_targets, mask_band):
            box_poses = attack_targets[0].bbox.cpu().detach().numpy()
            batch, channel, width, height = input.size()
            grad_mask = torch.zeros((width, height)).to(input.device)
            
            for box in box_poses:
                box_center = ( (int(box[1])+int(box[3]))//2, (int(box[0])+int(box[2]))//2)
                grad_mask[int(box[1]):int(box[3]), box_center[1] - mask_band: box_center[1] + mask_band] = 1
                grad_mask[box_center[0] - mask_band: box_center[0] + mask_band, int(box[0]):int(box[2])] = 1
            
            box_poses = source_targets[0].bbox.cpu().detach().numpy()

            for box in box_poses:
                box_center = ( (int(box[1])+int(box[3]))//2, (int(box[0])+int(box[2]))//2)
                grad_mask[int(box[1]):int(box[3]), box_center[1] - mask_band: box_center[1] + mask_band] = 1
                grad_mask[box_center[0] - mask_band: box_center[0] + mask_band, int(box[0]):int(box[2])] = 1
            return grad_mask

    def cross_X_patch_select(self, input, attack_targets, source_targets, mask_band):
            box_poses = attack_targets[0].bbox.cpu().detach().numpy()
            batch, channel, width, height = input.size()
            grad_mask = torch.zeros((width, height)).to(input.device)
            for box in box_poses:
                width = int(box[3] - box[1])
                height = int(box[2] - box[0])

                max_value = max(width, height)
                for pos in range(max_value):
                        point_x = int(width/max_value * pos)
                        point_x2 =int( width - width/max_value * pos)
                        point_y = int(height/max_value * pos) + int(box[0])
                        min_point_x = max(0, point_x - mask_band) + int(box[1])
                        max_point_x = min(width, point_x + mask_band) + int(box[1])
                        min_point_x2 = max(0, point_x2 - mask_band) + int(box[1])
                        max_point_x2 = min(width, point_x2 + mask_band) +int(box[1])
                        grad_mask[min_point_x:max_point_x, point_y] = 255
                        grad_mask[min_point_x2:max_point_x2, point_y] = 255

            box_poses = source_targets[0].bbox.cpu().detach().numpy()

            for box in box_poses:
                width = int(box[3] - box[1])
                height = int(box[2] - box[0])

                max_value = max(width, height)
                for pos in range(max_value):
                        point_x = int(width/max_value * pos)
                        point_x2 =int( width - width/max_value * pos)
                        point_y = int(height/max_value * pos) + int(box[0])
                        min_point_x = max(0, point_x - mask_band) + int(box[1])
                        max_point_x = min(width, point_x + mask_band) + int(box[1])
                        min_point_x2 = max(0, point_x2 - mask_band) + int(box[1])
                        max_point_x2 = min(width, point_x2 + mask_band) +int(box[1])
                        grad_mask[min_point_x:max_point_x, point_y] = 255
                        grad_mask[min_point_x2:max_point_x2, point_y] = 255

            return grad_mask

    def rec_patch_select(self, input, attack_targets, source_targets, mask_band): 

            box_poses = attack_targets[0].bbox.cpu().detach().numpy()
            batch, channel, width, height = input.size()
            grad_mask = torch.zeros((width, height)).to(input.device)
            for box in box_poses:
                grad_mask[int(box[1]):int(box[3]), int(box[0]) :int(box[2])] = 1
            box_poses = source_targets[0].bbox.cpu().detach().numpy()
            for box in box_poses:
                grad_mask[int(box[1]):int(box[3]), int(box[0]) :int(box[2])] = 1
            return grad_mask




class MomentumIteratorAttack(Attack):

    def __init__(self, model):

        super(MomentumIteratorAttack, self).__init__(model)

    def __call__(self, input,*args,  **kwargs):
        return self._apply( input,*args,  **kwargs)

    def _apply(self, images,targets,
               normal, model_ls,
               attack_mode,
               norm_ord='inf',
               epsilons=0.05,
               epsilons_max=0.3,
               steps=20,
               epsilon_steps=5,
               min_=0, max_ =  255,decay_factor=1,clip_each_iter = False):

        if norm_ord == 0:
            raise ValueError("L0 norm is not supported!")

        logging.info('epsilons={0},epsilons_max={1},steps={2},epsilon_steps={3}'.
                     format(epsilons,epsilons_max,steps,epsilon_steps))

        if not isinstance(epsilons, Iterable):
            #从epsilons到0.5逐步增大
            epsilons = np.linspace(epsilons, epsilons_max, num=epsilon_steps)

        #从[epsilon,0.5]动态调整epsilon 直到攻击成功
        step = 0
        input = images.tensors
        adv_img = input
        attack_targets = targets[0]
        GT_triplets = self.get_GT_triplet(attack_targets)
        momentum = 0

        for epsilon in epsilons[:]:
            if epsilon == 0.0:
                continue
            for i in range(steps):

                adv_img = torch.tensor(adv_img, requires_grad=True)
                if get_loss.get_loss(self.model,adv_img,images, attack_targets, attack_mode) is None:
                    print('No proposal pairs')
                    return is_success, adv_img, proposals,proposals_pairs
                loss, proposals ,proposals_pairs= get_loss.get_loss(self.model,adv_img,images, attack_targets, attack_mode)
                gradient = - get_grad.get_grad(adv_img, loss, self.model)

                velocity = gradient / self._norm(gradient, ord=1)
                momentum = decay_factor * momentum + velocity
                if norm_ord == 'inf':
                    gradient_norm = torch.sign(momentum)
                else:
                    gradient_norm = self._norm(momentum, ord=norm_ord)
                adv_img = adv_img + epsilon * gradient_norm
                if clip_each_iter:
                    adv_img = self.recover_img(adv_img, normal, min_, max_)
                    adv_img = self.preprocess_img(adv_img,normal)
                is_success, proposals ,proposals_pairs = self.is_success(attack_mode,GT_triplets,adv_img, images, attack_targets)
                if is_success:
                    return is_success, adv_img, proposals,proposals_pairs

            step += 1
        return is_success, adv_img , proposals,proposals_pairs


class FastGradientSignMethodAttack(Attack):

    def __init__(self, model):

        super(FastGradientSignMethodAttack, self).__init__(model)

    def __call__(self, input,*args,  **kwargs):
        return self._apply( input,*args,  **kwargs)
    def _apply(self, images,targets,
               normal,model_ls,
               attack_mode,
               norm_ord='inf',
               epsilon=0.1,
               min_=0, max_ =  255,clip_each_iter = False):

        if norm_ord == 0:
            raise ValueError("L0 norm is not supported!")

        input = images.tensors
        adv_img = input
        targets = targets[0]
        epsilon = epsilon * (max_ - min_)
        adv_img = torch.tensor(adv_img, requires_grad=True)
        if get_loss.get_loss(self.model,adv_img,images, targets, attack_mode) is None:
            print('No proposal pairs')
            return is_success, adv_img, ob_prop,proposals_pairs
        loss, ob_prop ,proposals_pairs= get_loss.get_loss(self.model,adv_img,images, targets, attack_mode)
        gradient = - get_grad.get_grad(adv_img, loss, self.model)

        if norm_ord == 'inf':
            gradient_norm = np.sign(gradient)
        else:
            gradient_norm = self._norm(gradient, ord=norm_ord)
        adv_img = adv_img + epsilon * gradient_norm
        is_success = self.is_success(adv_img, attack_mode)
        if clip_each_iter:
            adv_img = self.recover_img(adv_img, normal, min_, max_)
            adv_img = self.preprocess_img(adv_img,normal)

        return is_success, adv_img , ob_prop,proposals_pairs

class BasicTest(Attack):

    def __init__(self, model):

        super(BasicTest, self).__init__(model)
        self.initial_perturbation = False
        self.init_weight = 0
        self.use_grad_mask = False
        self.grad_mask = None
        self.dynamic_select = False
        self.dynamic_ratio = 1
        self.mask_band = 0
        self.ori_mask = False
    def __call__(self, input,*args,  **kwargs):
        return self._apply( input,*args,  **kwargs)
    def _apply(self, images,targets,
               normal,model_ls,
               attack_mode,
               norm_ord='inf',
               epsilons=0.1,
               epsilons_max=1.0,
               budget = 0.1,
               steps=20,
               epsilon_steps=5, 
               min_=0, max_ =  255,
                is_validate = False,clip_each_iter = False):
        # print(self.model.name)
        if norm_ord == 0:
            raise ValueError("L0 norm is not supported!")

        logging.info('epsilons={0},epsilons_max={1},steps={2},epsilon_steps={3}'.
                     format(epsilons,epsilons_max,steps,epsilon_steps))

        if not isinstance(epsilons, Iterable):
            #从epsilons到0.5逐步增大
            epsilons = np.linspace(epsilons, epsilons_max, num=epsilon_steps)

        #从[epsilon,0.5]动态调整epsilon 直到攻击成功

        input = images.tensors
        adv_img  = input
        attack_record = {'attack_mode': attack_mode, 'best_adv_img': adv_img, 'ar_ratio':[0,0,0,0,0],'best_rel_ratio': [0.,0.,0.,0.,0.], 'best_obj_ratio': [0.,0.,0.,0.,0.]}
        attack_targets = targets[0]
        GT_triplets = self.get_GT_triplet(attack_targets)
        adv_img = adv_img.clone().detach()
        adv_img = adv_img.squeeze()*normal[1] + normal[0]
        import cv2
        per_ = cv2.imread('./24.png')
        per_ = torch.tensor(per_).to(adv_img.device).unsqueeze(0)
        per_ = per_.permute((0,3, 1, 2)).contiguous()  
        per_ = per_[:,[ 2,1,0]]
        adv_img = torch.clamp(adv_img + per_ , min_, max_)
        adv_img = (adv_img - normal[0])/normal[1]

        is_success, updated_flag, proposals ,proposals_pairs = self.is_success(attack_record,model_ls,GT_triplets,adv_img, images, attack_targets)
        attack_record['best_adv_img'] = adv_img

        return is_success, attack_record, proposals,proposals_pairs



GCPA = Gradient_CrossPatch_Attack
GRPA = Gradient_RecPatch_Attack
BIM = BasicIterativeMethodAttack
MIFGSM = MomentumIteratorAttack
FGSM = FastGradientSignMethodAttack
SBIM = SparsityBasicIterativeMethodAttack
SBIM2 = SparsityBasicIterativeMethodAttack2
PGD = ProjectGradientDAttack
GCXPA = Gradient_CrossXPatch_Attack
B_Test = BasicTest
