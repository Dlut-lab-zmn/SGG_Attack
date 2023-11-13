# Copyright (c) Mn,Zhao. All Rights Reserved.
from builtins import object
import logging
from abc import ABCMeta
from abc import abstractmethod
from unittest import result
import get_loss
import numpy as np
from future.utils import with_metaclass

import torch
class Attack(with_metaclass(ABCMeta, object)):
    """
    Abstract base class for adversarial attacks. `Attack` represent an
    adversarial attack which search an adversarial example. subclass should
    implement the _apply() method.
    Args:
        model(Model): an instance of the class adversarialbox.base.Model.
    """

    def __init__(self, model):
        self.model = model

    def __call__(self, adversary,*args,  **kwargs):
        """
        Generate the adversarial sample.
        Args:
        adversary(object): The adversary object.
        **kwargs: Other named arguments.
        """
        self._preprocess(adversary)
        return self._apply(adversary, *args, **kwargs)

    @abstractmethod
    def _apply(self, adversary, *args,**kwargs):
        """
        Search an adversarial example.
        Args:
        adversary(object): The adversary object.
        **kwargs: Other named arguments.
        """
        raise NotImplementedError

    def evaluate(self, attack_record, ind, model,*args):

        attack_mode = self.get_attack_mode(attack_record['attack_mode'])
        GT_triplets,GT_objs, result = get_loss.get_pred(model, *args)
        if result is not None:
            loss, proposals ,proposals_pairs = result
        else:
            return None
        ob_prop_value, ob_prop_class = torch.max(proposals[0].get_field('scores_all'), 1)
        ob_prop_index = ob_prop_class.nonzero().squeeze()
        ob_prop_class = ob_prop_class.cpu().detach().numpy()
        pred_labels = proposals_pairs[0].get_field('labels').cpu().detach().numpy()
        idx_pairs = proposals_pairs[0].get_field('idx_pairs').cpu().detach().numpy()
        sub_idx_pairs = idx_pairs[:, 0]
        ob_idx_pairs = idx_pairs[:, 1]
        index_pairs = []
        triplets = []
        for i in range(len(sub_idx_pairs)):
            if sub_idx_pairs[i] in ob_prop_index and ob_idx_pairs[i] in ob_prop_index:
                index_pairs.append(i)
                triplets.append((ob_prop_class[sub_idx_pairs[i]], pred_labels[i],ob_prop_class[ob_idx_pairs[i]]))
        is_rel_success = True
        is_obj_success = True
        nor_rel_samples = 0
        nor_obj_samples = 0
        for GT_triplet  in GT_triplets:
            if attack_mode[0]:
                if GT_triplet not in triplets:
                    is_rel_success = False
                    nor_rel_samples += 1 
            if attack_mode[1]:
                if GT_triplet in triplets:
                    is_rel_success = False
                    nor_rel_samples += 1 
        for GT_obj in GT_objs:
            if attack_mode[0]:
                if GT_obj not in ob_prop_class:
                    nor_obj_samples += 1
                    is_obj_success = False
            if attack_mode[1]:
                if GT_obj in ob_prop_class:
                    nor_obj_samples += 1
                    is_obj_success = False
        is_success = is_rel_success & is_obj_success
        if is_success:
            attack_record['ar_ratio'][ind] = 1
        adv_rel_ratio = (len(GT_triplets) - nor_rel_samples)/len(GT_triplets)
        adv_obj_ratio = (len(GT_objs) - nor_obj_samples)/len(GT_objs)
        updated_flag = False
        #if adv_rel_ratio  > attack_record['best_rel_ratio'][ind] :
        #    attack_record['best_rel_ratio'][ind] = adv_rel_ratio
        #if adv_obj_ratio > attack_record['best_obj_ratio'][ind]:
        #    attack_record['best_obj_ratio'][ind] = adv_obj_ratio
        if adv_rel_ratio * adv_obj_ratio > attack_record['best_rel_ratio'][ind] * attack_record['best_obj_ratio'][ind]:
            attack_record['best_rel_ratio'][ind] = adv_rel_ratio
            attack_record['best_obj_ratio'][ind] = adv_obj_ratio
            updated_flag = True
        
        return is_success, updated_flag, proposals ,proposals_pairs

    def is_success(self, attack_record, model_ls,*args):
        for i in range(len(model_ls)-1):
            if model_ls[-1] == i:
                evaluate_result= self.evaluate(attack_record, i, model_ls[i], *args)
            else:
                self.evaluate(attack_record, i, model_ls[i], *args)
        return evaluate_result#is_success, updated_flag, proposals ,proposals_pairs

    def get_attack_mode(self, attack_mode):
        attack = [False,False]
        if len(attack_mode[0])>0:
            attack[0] = True
        if len(attack_mode[1])>0:
            attack[1] = True
        return attack
    def get_GT_triplet(self, targets):
        cpu_target_label = targets[0].get_field("labels").cpu().detach().numpy()
        cpu_pred_label = targets[0].get_field("pred_labels").cpu().detach().numpy()

        cpu_pred_index = cpu_pred_label.nonzero()
        start_pred_index = cpu_pred_index[0]
        end_pred_index = cpu_pred_index[1]
        GT_triplets = []
        for i in range(len(start_pred_index)):
            GT_triplets.append((cpu_target_label[start_pred_index[i]], cpu_pred_label[start_pred_index[i],end_pred_index[i]],cpu_target_label[end_pred_index[i]]))
        return GT_triplets

    def recover_img(self, adv_img, normal, min_, max_):
            adv_img = adv_img.squeeze()*normal[1] + normal[0]
            adv_img = adv_img.permute((1,2,0)).contiguous()#[[2,1,0]]
            adv_img = torch.clamp(adv_img, min_, max_)
            return adv_img
    def preprocess_img(self, adv_img, normal):
            adv_img = adv_img.permute((2, 0, 1)).contiguous()  
            adv_img = (adv_img - normal[0])/normal[1]#[[2,1,0]]
            return adv_img.unsqueeze(0)
    @staticmethod
    def _norm(a, ord):
        if a.ndim == 3:
            dim = (1, 2)
            reshape = (a.shape[0], 1, 1)
            weight = a.shape[1]*a.shape[2]
        elif a.ndim == 4:
            dim = (0, 2, 3)
            reshape = (1, a.shape[1], 1, 1)
            weight = a.shape[2]*a.shape[3]
        else:
            logging.raiseExceptions('a.shape should 1, 3, 4')
        return torch.norm(a, ord, dim).view(reshape) / weight


    def _preprocess(self, adversary):
        """
        Preprocess the adversary object.
        :param adversary: adversary
        :return: None
        """
        #assert self.model.channel_axis() == adversary.original.ndim

        if adversary.original_label is None:
            adversary.original_label = np.argmax(
                self.model.predict(adversary.original))
        if adversary.is_targeted_attack and adversary.target_label is None:
            if adversary.target is None:
                raise ValueError(
                    'When adversary.is_targeted_attack is true, '
                    'adversary.target_label or adversary.target must be set.')
            else:
                adversary.target_label = np.argmax(
                    self.model.predict(adversary.target))

        logging.info('adversary:'
                     '\n         original_label: {}'
                     '\n         target_label: {}'
                     '\n         is_targeted_attack: {}'
                     ''.format(adversary.original_label, adversary.target_label,
                               adversary.is_targeted_attack))
