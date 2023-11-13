# Copyright (c) Mn,Zhao. All Rights Reserved.
import torch
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist_with_fields
from od_adv_box.models import Motifs as base_model
def reldn(model,input, images , targets, only_OD =True, is_train =True):
    _, proposals, class_logits, box_regression, loss_dict_ob = base_model.object_detector(model,input, images , targets, is_train = is_train)

    if only_OD:
        proposals_pairs = None
        return loss_dict_ob,proposals, proposals_pairs
    else:
        features = model.rel_backbone(input) # reldn
        relation_result = relation_head(model, features, proposals, targets, is_train)
        if relation_result is None:
            return None
        x_pairs,pred_class_logits, proposals_pairs, loss_dict_rel = relation_result
        loss_dict_ob.update(loss_dict_rel)
        return loss_dict_ob, proposals, proposals_pairs



def relation_head(model, features, proposals, targets,is_train):
    #if is_train:
    #    NUM_OBJS = len(torch.unique(targets[0].get_field("pred_labels").nonzero()))
    #    proposals = model.relation_head.loss_evaluator.sel_proposals(proposals,NUM_OBJS)
    #print(proposals)

    obj_features = model.obj_feature_extractor(features, proposals, use_relu=False)
    if obj_features.ndimension() == 4:
        obj_features = torch.nn.functional.adaptive_avg_pool2d(obj_features, 1)
        obj_features = obj_features.view(obj_features.size(0), -1)
    boxes_per_image = [len(box) for box in proposals]
    obj_features = obj_features.split(boxes_per_image, dim=0)
    for proposal, obj_feature in zip(proposals, obj_features):
        proposal.add_field('box_features', obj_feature)


    subj_features = model.subj_feature_extractor(features, proposals, use_relu=False)
    if subj_features.ndimension() == 4:
        subj_features = torch.nn.functional.adaptive_avg_pool2d(subj_features, 1)
        subj_features = subj_features.view(subj_features.size(0), -1)
    boxes_per_image = [len(box) for box in proposals]
    subj_features = subj_features.split(boxes_per_image, dim=0)
    for proposal, subj_feature, obj_feature in zip(proposals, subj_features, obj_features):
        proposal.add_field('subj_box_features', subj_feature)
        proposal.add_field('obj_box_features', obj_feature)
    
    if is_train:
        gt_features = model.obj_feature_extractor(features, targets, use_relu=False)
        if gt_features.ndimension() == 4:
            gt_features = torch.nn.functional.adaptive_avg_pool2d(gt_features, 1)
            gt_features = gt_features.view(gt_features.size(0), -1)
        gt_boxes_per_image = [len(box) for box in targets]
        assert sum(gt_boxes_per_image)==len(gt_features), "gt_boxes_per_image and len(gt_features) do not match!"
        gt_features = gt_features.split(gt_boxes_per_image, dim=0)
        for target, gt_feature in zip(targets, gt_features):
            target.add_field('box_features', gt_feature)
            target.add_field('gt_labels', target.get_field('labels'))

        gt_subj_features = model.subj_feature_extractor(features, targets, use_relu=False)
        if gt_subj_features.ndimension() == 4:
            gt_subj_features = torch.nn.functional.adaptive_avg_pool2d(gt_subj_features, 1)
            gt_subj_features = gt_subj_features.view(gt_subj_features.size(0), -1)
        gt_boxes_per_image = [len(box) for box in targets]
        gt_subj_features = gt_subj_features.split(gt_boxes_per_image, dim=0)
        for target, gt_subj_feature, gt_feature in zip(targets, gt_subj_features, gt_features):
            target.add_field('subj_box_features', gt_subj_feature)
            target.add_field('obj_box_features', gt_feature)
        # contrastive_loss_sample
        # suj Pos       obj Pos      pair Pos
        # suj Pos       obj Pos      pair Neg
        # suj Pos       obj Neg      pair Neg
        # suj Neg       obj Pos      pair Neg

        proposal_pairs = model.relation_head.loss_evaluator.contrastive_loss_sample(model.cfg, proposals, targets)
        fields = ['box_features', 'labels', 'subj_box_features', 'obj_box_features']

        proposals = [cat_boxlist_with_fields([proposal_per_image, target], fields) for proposal_per_image, target in zip(proposals, targets)]
        model.relation_head.loss_evaluator.contrastive_proposal_pair_transform(proposals, proposal_pairs)

    else:
        proposal_pairs = model.relation_head._get_proposal_pairs(proposals)
    if proposal_pairs is None:
        return None
    if proposal_pairs[0].bbox.shape[0] == 0:
        return None
    x, obj_class_logits, pred_class_logits, obj_class_preds, rel_inds \
    = model.relation_head.rel_predictor(features, proposals, proposal_pairs) ##proposal_pair_features, obj distribution, relation_class, obj_class, relation_index
    pred_class_logits = pred_class_logits + model.relation_head.freq_bias.index_with_labels(
                                                                                                                                                                                    torch.stack((
                                                                                                                                                                                    obj_class_preds[rel_inds[:, 0]],
                                                                                                                                                                                    obj_class_preds[rel_inds[:, 1]],
                                                                                                                                                                                    ), 1) ).cuda()

    if not is_train:
        proposal_pairs = model.relation_head.post_processor(pred_class_logits, proposal_pairs, x,
                                        use_freq_prior=model.relation_head.cfg.MODEL.USE_FREQ_PRIOR)
        proposals, proposal_pairs = base_model._post_processing_constrained(model,proposals, proposal_pairs)
        return x,pred_class_logits, proposal_pairs, {}

    loss_obj_classifier = torch.tensor(0, dtype=torch.float).to(pred_class_logits.device)
    loss_pred_classifier = model.relation_head.loss_evaluator.cross_entropy_losses([pred_class_logits], assign_weight =1)

    # CONTRASTIVE_LOSS.USE_NODE_CONTRASTIVE_LOSS:
    loss_contrastive_sbj, loss_contrastive_obj = model.relation_head.loss_evaluator.reldn_contrastive_losses(model.cfg, [pred_class_logits])
    loss_contrastive_sbj = loss_contrastive_sbj * model.cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.NODE_CONTRASTIVE_WEIGHT
    loss_contrastive_obj = loss_contrastive_obj * model.cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.NODE_CONTRASTIVE_WEIGHT
    # USE_NODE_CONTRASTIVE_SO_AWARE_LOSS:
    loss_so_contrastive_sbj, loss_so_contrastive_obj = model.relation_head.loss_evaluator.reldn_so_contrastive_losses(model.cfg, [pred_class_logits])
    loss_so_contrastive_sbj = loss_so_contrastive_sbj * model.cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.NODE_CONTRASTIVE_SO_AWARE_WEIGHT
    loss_so_contrastive_obj = loss_so_contrastive_obj * model.cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.NODE_CONTRASTIVE_SO_AWARE_WEIGHT
    # USE_NODE_CONTRASTIVE_P_AWARE_LOSS:
    loss_p_contrastive_sbj, loss_p_contrastive_obj = model.relation_head.loss_evaluator.reldn_p_contrastive_losses(model.cfg, [pred_class_logits])
    loss_p_contrastive_sbj = loss_p_contrastive_sbj * model.cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.NODE_CONTRASTIVE_P_AWARE_WEIGHT
    loss_p_contrastive_obj = loss_p_contrastive_obj * model.cfg.MODEL.ROI_RELATION_HEAD.CONTRASTIVE_LOSS.NODE_CONTRASTIVE_P_AWARE_WEIGHT

    return (
        x,pred_class_logits,
        proposal_pairs,
        # Reldn_OB_Cls  Reldn_Pred_Cls   Reldn_Agnostic_Sub_Cls  Reldn_Agnostic_Obj_Cls  Reldn_Eca_Sub_Cls Reldn_Eca_Obj_Cls  Reldn_Pca_Sub_Cls  Reldn_Pca_obj_Cls
        dict(Reldn_OB_Cls=loss_obj_classifier, Reldn_Pred_Cls=loss_pred_classifier, \
            Reldn_Agnostic_Sub_Cls=loss_contrastive_sbj, Reldn_Agnostic_Obj_Cls=loss_contrastive_obj, \
            Reldn_Eca_Sub_Cls=loss_so_contrastive_sbj, Reldn_Eca_Obj_Cls=loss_so_contrastive_obj, \
            Reldn_Pca_Sub_Cls=loss_p_contrastive_sbj, Reldn_Pca_obj_Cls=loss_p_contrastive_obj),
    )
