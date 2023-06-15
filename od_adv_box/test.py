import argparse
import torch.optim as optim
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
from psnr import psnr
import torch
from maskrcnn_benchmark.config import cfg_motifs,cfg_motifs2, cfg_reldn, cfg_imp, cfg_msdn, cfg_grcnn
from models.generator import  GeneratorResnet
from scene_graph_benchmark.config import sg_cfg
from maskrcnn_benchmark.utils.comm import synchronize
import load_data
import od_adv_box.load_models as load_models
import cv2
from gradient_attack import USGUA
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
    elif 'mn' in config_file:
        cfg = cfg_motifs2
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

    base_path = './sgg_configs/vg_vrd/rel_danfeiX_FPN50_'
    cfg_nm = load_cfg(base_path + 'nm.yaml')
    cfg_reldn = load_cfg(base_path +  'reldn.yaml')
    cfg_imp = load_cfg(base_path + 'imp.yaml')
    cfg_msdn = load_cfg(base_path + 'msdn.yaml')
    cfg_grcnn = load_cfg(base_path + 'grcnn.yaml')
    #device = torch.device(cfg_nm.MODEL.DEVICE)
    if 'Motifs' in cfg_nm.MODEL_REL:
        Pred_Cls = 'Motifs_Pred_Cls'
        cfg = cfg_nm
        model = load_models.load_models(cfg, 'Motifs')
    if 'Reldn' in cfg_nm.MODEL_REL:
        Pred_Cls = 'Reldn_Pred_Cls'
        cfg = cfg_reldn
        model = load_models.load_models(cfg, 'Reldn')
    if 'Imp' in cfg_nm.MODEL_REL:
        Pred_Cls = 'Imp_Pred_Cls'
        cfg = cfg_imp
        model = load_models.load_models(cfg, 'Imp')
    if 'Msdn' in cfg_nm.MODEL_REL:
        Pred_Cls = 'Msdn_Pred_Cls'
        cfg = cfg_msdn
        model = load_models.load_models(cfg, 'Msdn')
    if 'Grcnn' in cfg_nm.MODEL_REL:
        Pred_Cls = 'Grcnn_Pred_Cls'
        cfg = cfg_grcnn
        model = load_models.load_models(cfg, 'Grcnn')
    output_folders, dataset_names, data_loaders_val, labelmap_file = load_data.load_data(cfg)
    model.train()
    netG = GeneratorResnet(data_dim='low')
    netG.cuda()
    netG.train()
    optimG = optim.Adam(netG.parameters(), lr=0.001, betas=(0.5, 0.999))
    gm_attack = USGUA(model,netG)
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
    model_ls = [model,0]
    # object detector: RPN_Bi   RPN_Proposal   OD_Cls   OD_Proposal 
    # motifs :  Motifs_OB_Cls   Motifs_Pred_Cls 
    # reldn:  Reldn_OB_Cls  Reldn_Pred_Cls   Reldn_Agnostic_Sub_Cls  Reldn_Agnostic_Obj_Cls  Reldn_Eca_Sub_Cls Reldn_Eca_Obj_Cls  Reldn_Pca_Sub_Cls  Reldn_Pca_obj_Cls
    # grcnn:  Grcnn_OB_Cls  Grcnn_Relpn_Cls  Grcnn_Pred_Cls
    # imp: Imp_OB_Cls   Imp_Pred_Cls
    # msdn: Msdn_OB_Cls   Msdn_Pred_Cls
    # Norm_ord: inf , 1 , 2

    Pos_ob_mode = ['RPN_Bi','RPN_Proposal','OD_Cls']  # , ,'RPN_Proposal','OD_Proposal'
    Pos_rel_mode = [cfg.MODEL_REL] #cfg_nm.MODEL_REL  # 'Reldn_Pred_Cls'
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

    torch.save(netG.state_dict(), "./models/models/" + str(0) + ".tar" )
    netG.load_state_dict(torch.load("./models/models/" + str(0) + ".tar"))

    with alive_bar(len(data_loaders_val)) as bar:
        for epsilon_iter in range(400):
            bar()
            for iteration, data_batch in enumerate(data_loaders_val):

                images, current_targets, image_ids, scales = data_batch[0], data_batch[1], data_batch[2], data_batch[3:]

                FLAG = True
                start_iteration = 0 # 
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
                    #save_input = gm_attack.recover_img(images.tensors, normal, min_, max_).cpu().detach().numpy() 
                    #save_ob_input = dipict_obj_img(save_input.copy(), targets, id_to_label)
                    #cv2.imwrite(os.path.join('./od_adv_box/obj_samples/ori_samples/',  str(iteration) + '.png'), save_ob_input)

                    #save_pred_input = dipict_pred_img(save_input.copy(), targets, id_to_label,idx_to_predicate, 'GT')
                    #cv2.imwrite(os.path.join('./od_adv_box/pred_samples/ori_samples/',  str(iteration) + '.png'), save_pred_input)

                    if is_ob_attack:
                        adv_img, save_perturb,proposals,proposals_pairs  =    gm_attack( images,targets,model_ls,normal,optimG,attack_mode = ob_attack_mode,beta = 0.01)

                    if is_relation_attack:
                        if is_ob_attack:
                            images.tensors = adv_img
                        adv_img, save_perturb,proposals,proposals_pairs = gm_attack( images,targets,model_ls,normal,optimG,attack_mode = rel_attack_mode,beta = 0.01)
                        save_adv_input = gm_attack.recover_img(adv_img, normal, min_, max_).cpu().detach().numpy() 
                        if len(save_adv_input.shape) == 3:
                            save_adv_input = np.expand_dims(save_adv_input,0)

                        if proposals_pairs is not None:
                            for i in range(cfg.TEST.IMS_PER_BATCH):

                                if len(proposals[i].bbox) != 0:
                                    ob_prop_value, ob_prop_class = torch.max(proposals[i].get_field('scores_all'), 1)
                                    ob_prop_index = ob_prop_class.nonzero().squeeze()
                                    ob_bbox2 = proposals[i].bbox


                                    proposals_pairs[i].bbox = ob_bbox2
                                    pred_labels = proposals_pairs[i].get_field('labels')
                                    idx_pairs = proposals_pairs[i].get_field('idx_pairs')
                                    sub_idx_pairs = idx_pairs[:, 0]
                                    ob_idx_pairs = idx_pairs[:, 1]
                                    index_pairs = []
                                    for j in range(len(sub_idx_pairs)):
                                        if sub_idx_pairs[j] in ob_prop_index and ob_idx_pairs[j] in ob_prop_index:
                                            index_pairs.append(j)
                                    pred_labels = pred_labels[index_pairs]
                                    idx_pairs = idx_pairs[index_pairs]
                                    proposals_pairs[i].add_field('pred_labels', pred_labels)
                                    proposals_pairs[i].add_field('labels', ob_prop_class)
                                    proposals_pairs[i].add_field('idx_pairs', idx_pairs)
                                    save_adv_input[i] = dipict_pred_img(save_adv_input[i].copy(), [proposals_pairs[i]], id_to_label,idx_to_predicate , 'Pred')
                                cv2.imwrite(os.path.join('./od_adv_box/pred_samples/adv_samples/',  str(iteration) + '_' + str(i) +  '.png'), save_adv_input[i])
        if epsilon_iter%100 == 0:
            torch.save(netG.state_dict(), "./models/models/" + str(epsilon_iter) + ".tar" )



