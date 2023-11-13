import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
from psnr import psnr
import torch
from maskrcnn_benchmark.config import cfg_motifs,cfg_motifs2, cfg_motifs3,cfg_reldn, cfg_imp, cfg_msdn, cfg_grcnn
from scene_graph_benchmark.config import sg_cfg
from maskrcnn_benchmark.utils.comm import synchronize
import load_data
import od_adv_box.load_models as load_models
import cv2
from gradient_attack import GCPA, GRPA, BIM, MIFGSM, FGSM , SBIM, PGD, SBIM2,B_Test
from func import id_to_name, dipict_obj_img, dipict_pred_img, modify_target_force_box, construct_graph, dipict_pred_img2
from maskrcnn_benchmark.structures.bounding_box import BoxList
from alive_progress import alive_bar
def load_cfg(config_file):
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default=config_file,
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--model-rel",
        help="Select the target model",
        default='Imp_Pred_Cls',
        type=str,
        
    )
    
    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed =  False#num_gpus > 1


    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    if 'nm' in config_file:
        cfg = cfg_motifs
    elif 'mn2' in config_file:
        cfg = cfg_motifs2
    elif 'mn3' in config_file:
        cfg = cfg_motifs3
    elif 'reldn' in config_file:
        cfg = cfg_reldn
    elif 'imp' in config_file:
        cfg = cfg_imp
    elif 'msdn' in config_file:
        cfg = cfg_msdn
    elif 'grcnn' in config_file:
        cfg = cfg_grcnn

    cfg.distributed = args.distributed


    cfg.set_new_allowed(True)

    cfg.merge_from_other_cfg(sg_cfg)

    cfg.set_new_allowed(False)
    cfg.MODEL_REL = args.model_rel
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if cfg.MODEL.ROI_RELATION_HEAD.MODE  == 'sgcls':
        cfg.MODEL.ROI_BOX_HEAD.FORCE_BOXES = True
    cfg.freeze()
    return cfg



if __name__ == "__main__":
    #if 'oi' in cfg.MODEL.WEIGHT:
    
    base_path = './sgg_configs/vg_vrd/rel_danfeiX_FPN50_'
    #base_path = './sgg_configs/oi_vrd/R152FPN_'
    cfg_nm = load_cfg(base_path + 'nm.yaml')
    #cfg_nm2 = load_cfg(base_path + 'mn2.yaml')
    #cfg_nm3 = load_cfg(base_path + 'mn3.yaml')
    cfg_reldn = load_cfg(base_path +  'reldn.yaml')
    cfg_imp = load_cfg(base_path + 'imp.yaml')
    cfg_msdn = load_cfg(base_path + 'msdn.yaml')
    cfg_grcnn = load_cfg(base_path + 'grcnn.yaml')
    model_names = ['Motifs','Reldn', 'Imp', 'Msdn', 'Grcnn']#, 'Grcnn']#,'Reldn', 'Imp']#, 'Msdn', 'Grcnn']# 
    cfg_ls = [cfg_nm,cfg_reldn,cfg_imp,cfg_msdn,cfg_grcnn]#cfg_grcnn]#cfg_reldn,cfg_imp]#,cfg_msdn,cfg_grcnn]#
    for model_ind, model_name in enumerate(model_names):
        if model_name in cfg_nm.MODEL_REL:
            break

    cfg = cfg_ls[model_ind]
    output_folders, dataset_names, data_loaders_val, labelmap_file = load_data.load_data(cfg)
    model_ind = 0
    model_nm = load_models.load_models(cfg_nm, 'Motifs')
    #model_mn2 = load_models.load_models(cfg_nm2, 'Motifs2')
    #model_mn3 = load_models.load_models(cfg_nm3, 'Motifs3')
    # model_reldn = load_models.load_models(cfg_reldn, 'Reldn')
    #model_imp = load_models.load_models(cfg_imp, 'Imp')
    #model_msdn = load_models.load_models(cfg_msdn, 'Msdn')
    # model_grcnn = load_models.load_models(cfg_grcnn, 'Grcnn')
    model_nm.train()
    #model_mn2.train()
    #model_mn3.train()
    # model_reldn.train()
    #model_imp.train()
    #model_msdn.train()
    # model_grcnn.train()
    model_ls = [model_nm, model_ind]#model_mn2, model_mn3, model_ind]#model_nm,model_reldn, model_imp, model_msdn, model_grcnn
    model = model_ls[model_ind]
    gm_attack = BIM(model)
    mean = cfg.INPUT.PIXEL_MEAN
    std=cfg.INPUT.PIXEL_STD
    mean= torch.as_tensor(mean, dtype=torch.float32)
    std= torch.as_tensor(std, dtype=torch.float32)
    mean = mean.view(-1, 1, 1).cuda()
    std = std.view(-1, 1, 1).cuda()
    normal = (mean, std)
    min_=0
    max_ =  255
    id_to_label, idx_to_predicate = id_to_name(cfg)
    
    # object detector: RPN_Bi   RPN_Proposal   OD_Cls   OD_Proposal 
    # motifs :  Motifs_OB_Cls   Motifs_Pred_Cls 
    # reldn:  Reldn_OB_Cls  Reldn_Pred_Cls   Reldn_Agnostic_Sub_Cls  Reldn_Agnostic_Obj_Cls  Reldn_Eca_Sub_Cls Reldn_Eca_Obj_Cls  Reldn_Pca_Sub_Cls  Reldn_Pca_obj_Cls
    # grcnn:  Grcnn_OB_Cls  Grcnn_Relpn_Cls  Grcnn_Pred_Cls
    # imp: Imp_OB_Cls   Imp_Pred_Cls
    # msdn: Msdn_OB_Cls   Msdn_Pred_Cls
    # Norm_ord: inf , 1 , 2

    Pos_ob_mode = ['RPN_Bi','RPN_Proposal','OD_Cls']  # , ,'RPN_Proposal','OD_Proposal'
    Pos_rel_mode = [cfg.MODEL_REL] #cfg.MODEL_REL
    Neg_ob_mode = []
    Neg_rel_mode = []
    ob_attack_mode = [Pos_ob_mode, Neg_ob_mode]  # [ [] for targeted,    [] for untargeted ]  
    rel_attack_mode = [Pos_ob_mode + Pos_rel_mode, Neg_ob_mode + Neg_rel_mode]  # [ [] for targeted,    [] for untargeted ]  
    is_ob_attack = False
    is_relation_attack = False
    is_targetd_attack = False
    is_untargeted_attack = False
    if len(Pos_ob_mode + Pos_rel_mode) > 0:
        is_targetd_attack = True
    if len(Neg_ob_mode + Neg_rel_mode) > 0:
        is_untargeted_attack = True
    if len(Pos_ob_mode + Neg_ob_mode) >0:
        is_ob_attack = True
    for mode_ in Pos_rel_mode + Neg_rel_mode :
        if model.name in mode_:
            is_relation_attack = True
    # 假设需要执行100个任务


    num_samples = 0
    sum_PSNR = 0
    sum_AR_nm = 0
    sum_AR_reldn = 0
    sum_AR_imp = 0
    sum_AR_msdn = 0
    sum_AR_grcnn = 0
    sum_rel_RATIO_nm = 0
    sum_rel_RATIO_reldn = 0
    sum_rel_RATIO_imp = 0
    sum_rel_RATIO_msdn = 0
    sum_rel_RATIO_grcnn = 0
    sum_obj_RATIO_nm = 0
    sum_obj_RATIO_reldn = 0
    sum_obj_RATIO_imp = 0
    sum_obj_RATIO_msdn = 0
    sum_obj_RATIO_grcnn = 0
    dict_triplet = {}
    dict_bbox = {}
    #import pickle

    #my_dict = {}
    


    with alive_bar(len(data_loaders_val)) as bar:
        for iteration, data_batch in enumerate(data_loaders_val):
            bar()
            images, current_targets, image_ids, scales = data_batch[0], data_batch[1], data_batch[2], data_batch[3:]


            FLAG = True
            start_iteration = 0
            if iteration < start_iteration:
                continue
            if is_targetd_attack:
                if iteration == start_iteration:
                    CONSTANT_TARGETS = current_targets
                    if is_relation_attack:
                        OB_LABELS = CONSTANT_TARGETS[0].get_field('labels')
                        REL_LABELS = CONSTANT_TARGETS[0].get_field('pred_labels')
                        OB_VALID_LABELS =  list(set(OB_LABELS.numpy()) - set(OB_LABELS[REL_LABELS.nonzero().reshape(-1)].numpy()) )
                        CONSTANT_TARGETS[0].add_field('ob_valid_labels',OB_VALID_LABELS)
                        pred_ind = CONSTANT_TARGETS[0].get_field('pred_labels').nonzero()
                        CONSTANT_PRED = CONSTANT_TARGETS[0].get_field('pred_labels')[pred_ind[:,0],pred_ind[:,1]]
                    else:
                        CONSTANT_TARGETS[0].add_field('ob_valid_labels',None)
                    continue
                targets = CONSTANT_TARGETS
                FLAG = current_targets[0].size ==targets[0].size
            else:
                current_targets[0].add_field('ob_valid_labels',None)
                targets = current_targets
            if  FLAG:
                if is_targetd_attack:
                    if cfg.MODEL.ROI_RELATION_HEAD.MODE == 'sgcls':
                        
                        find_flag = modify_target_force_box(current_targets, CONSTANT_TARGETS)
                        ob_labels = current_targets[0].get_field('labels')
                        pred_labels = current_targets[0].get_field('pred_labels')
                        rel_id_pairs =list(set(range(len(ob_labels))) -  set(pred_labels.nonzero().reshape(-1).numpy()))
                        ob_valid_labels =  ob_labels[rel_id_pairs]
                        if not find_flag:
                            continue
                        targets = current_targets
                        targets[0].add_field('ob_valid_labels',ob_valid_labels)

                images = images.to_cuda()
                targets = [target.to_cuda() for target in targets if target is not None]

                num_samples += 1
                targets[0].add_field('confis', torch.ones_like(targets[0].get_field('labels')))
                input = images.tensors.clone()
                save_input = gm_attack.recover_img(images.tensors, normal, min_, max_).cpu().detach().numpy() 
                save_ob_input = dipict_obj_img(save_input.copy(), targets, id_to_label)
                cv2.imwrite(os.path.join('./od_adv_box/obj_samples/ori_samples/',  str(iteration) + '.png'), save_ob_input)

                save_pred_input = dipict_pred_img(save_input.copy(), targets, id_to_label,idx_to_predicate, 'GT')
                cv2.imwrite(os.path.join('./od_adv_box/pred_samples/ori_samples/',  str(iteration) + '.png'), save_pred_input)
                is_ob_attack = False
                if is_ob_attack:
                    is_success, attack_record, ob_prop , proposals_pairs = gm_attack(images, (targets,current_targets), normal,model_ls,
                                                                                                                                                    attack_mode = ob_attack_mode, norm_ord = 'inf',
                                                                                                                                                        epsilons=0.1, epsilons_max=1,
                                                                                                                                                    steps=10, epsilon_steps=5, is_validate = True,
                                                                                                                                                    clip_each_iter = False)
                if is_relation_attack:
                    if is_ob_attack:
                        images.tensors = attack_record['best_adv_img']
                    is_success,attack_record, ob_prop , proposals_pairs = gm_attack(images, (targets,current_targets), normal,model_ls,
                                                                                                                                                    attack_mode = rel_attack_mode, norm_ord = 'inf',
                                                                                                                                                        epsilons=0.1, epsilons_max=1,
                                                                                                                                                    steps=20, epsilon_steps=5,is_validate = True,
                                                                                                                                                    clip_each_iter = False)

                adv_img = attack_record['best_adv_img']
                ob_prop_value, ob_prop_class = torch.max(ob_prop[0].get_field('scores_all'), 1)
                ob_prop_index = ob_prop_class.nonzero().squeeze()
                ob_bbox2 = ob_prop[0].bbox
                ob_prop[0].bbox = ob_bbox2[ob_prop_index]
                ob_prop_class2 = ob_prop_class[ob_prop_index]
                ob_prop_value2 = ob_prop_value[ob_prop_index]
                ob_prop[0].add_field('labels',ob_prop_class2)
                ob_prop[0].add_field('confis',ob_prop_value2)

                save_adv_obimg = gm_attack.recover_img(adv_img, normal, min_, max_).cpu().detach().numpy() 
                save_adv_input = dipict_obj_img(save_adv_obimg.copy(), ob_prop, id_to_label)
                cv2.imwrite(os.path.join('./od_adv_box/pred_samples/ori_samples/',  str(iteration) + '.png'), save_adv_input)

                proposals_pairs[0].bbox = ob_bbox2
                pred_labels = proposals_pairs[0].get_field('labels')
                idx_pairs = proposals_pairs[0].get_field('idx_pairs')
                sub_idx_pairs = idx_pairs[:, 0]
                ob_idx_pairs = idx_pairs[:, 1]
                index_pairs = []
                for i in range(len(sub_idx_pairs)):
                    if sub_idx_pairs[i] in ob_prop_index and ob_idx_pairs[i] in ob_prop_index:
                        index_pairs.append(i)
                pred_labels = pred_labels[index_pairs]
                idx_pairs = idx_pairs[index_pairs]
                proposals_pairs[0].add_field('pred_labels', pred_labels)
                proposals_pairs[0].add_field('labels', ob_prop_class)
                proposals_pairs[0].add_field('idx_pairs', idx_pairs)
                
                save_adv_predimg = gm_attack.recover_img(adv_img, normal, min_, max_).cpu().detach().numpy() 

                noise = np.abs(save_adv_predimg - save_input)
                save_adv_input = dipict_pred_img(save_adv_predimg.copy(), proposals_pairs, id_to_label,idx_to_predicate , 'Pred')
                cv2.imwrite(os.path.join('./od_adv_box/pred_samples/adv_samples/',  str(iteration) + '.png'), save_adv_predimg)
                cv2.imwrite(os.path.join('./od_adv_box/pred_samples/noise/',  str(iteration) + '.png'), noise)
                
                sum_PSNR += psnr(save_input, save_adv_predimg) 
                sum_AR_nm += attack_record['ar_ratio'][0]
                sum_AR_reldn += attack_record['ar_ratio'][1]
                sum_AR_imp += attack_record['ar_ratio'][2]
                sum_AR_msdn += attack_record['ar_ratio'][3]
                sum_AR_grcnn += attack_record['ar_ratio'][4]
                sum_rel_RATIO_nm += attack_record['best_rel_ratio'][0]
                sum_rel_RATIO_reldn += attack_record['best_rel_ratio'][1]
                sum_rel_RATIO_imp += attack_record['best_rel_ratio'][2]
                sum_rel_RATIO_msdn += attack_record['best_rel_ratio'][3]
                sum_rel_RATIO_grcnn += attack_record['best_rel_ratio'][4]
                sum_obj_RATIO_nm += attack_record['best_obj_ratio'][0]
                sum_obj_RATIO_reldn += attack_record['best_obj_ratio'][1]
                sum_obj_RATIO_imp += attack_record['best_obj_ratio'][2]
                sum_obj_RATIO_msdn += attack_record['best_obj_ratio'][3]
                sum_obj_RATIO_grcnn += attack_record['best_obj_ratio'][4]
                print("mean_PSNR:  %0.3f; \t mean_AR:  %0.3f; \t mean_rel_RATIO: %0.3f; \t mean_obj_RATIO: %0.3f"%(sum_PSNR/num_samples, sum_AR_nm/num_samples, sum_rel_RATIO_nm/num_samples, sum_obj_RATIO_nm/num_samples), num_samples)
                print("mean_PSNR:  %0.3f; \t mean_AR:  %0.3f; \t mean_rel_RATIO: %0.3f; \t mean_obj_RATIO: %0.3f"%(sum_PSNR/num_samples, sum_AR_reldn/num_samples, sum_rel_RATIO_reldn/num_samples, sum_obj_RATIO_reldn/num_samples), num_samples)
                print("mean_PSNR:  %0.3f; \t mean_AR:  %0.3f; \t mean_rel_RATIO: %0.3f; \t mean_obj_RATIO: %0.3f"%(sum_PSNR/num_samples, sum_AR_imp/num_samples, sum_rel_RATIO_imp/num_samples, sum_obj_RATIO_imp/num_samples), num_samples)
                print("mean_PSNR:  %0.3f; \t mean_AR:  %0.3f; \t mean_rel_RATIO: %0.3f; \t mean_obj_RATIO: %0.3f"%(sum_PSNR/num_samples, sum_AR_msdn/num_samples, sum_rel_RATIO_msdn/num_samples, sum_obj_RATIO_msdn/num_samples), num_samples)
                print("mean_PSNR:  %0.3f; \t mean_AR:  %0.3f; \t mean_rel_RATIO: %0.3f; \t mean_obj_RATIO: %0.3f"%(sum_PSNR/num_samples, sum_AR_grcnn/num_samples, sum_rel_RATIO_grcnn/num_samples, sum_obj_RATIO_grcnn/num_samples), num_samples)
                



