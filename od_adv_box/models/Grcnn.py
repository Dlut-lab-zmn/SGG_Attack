# Copyright (c) Mn,Zhao. All Rights Reserved.
import torch
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist_with_fields
from od_adv_box.models import Motifs as base_model
def grcnn(model,input, images , targets, only_OD =True, is_train =True):
    features, proposals, class_logits, box_regression, loss_dict_ob = base_model.object_detector(model,input, images , targets, is_train = is_train)
    if only_OD:
        proposals_pairs = None
        return loss_dict_ob,proposals, proposals_pairs
    else:
        if not model.cfg.MODEL.ROI_RELATION_HEAD.SHARE_CONV_BACKBONE:
            features = model.rel_backbone(images.tensors)
        else:
            features = [feature.detach() for feature in features]
        relation_result = relation_head(model, features, proposals, targets, is_train)
        if relation_result is None:
            return None
        x_pairs,pred_class_logits, proposals_pairs, loss_dict_rel = relation_result
        loss_dict_ob.update(loss_dict_rel)
        return loss_dict_ob, proposals, proposals_pairs

def relation_head(model, features, proposals, targets,is_train):
    # NUM_OBJS = len(torch.unique(targets[0].get_field("pred_labels").nonzero()))
    # proposals = model.relation_head.loss_evaluator.sel_proposals(proposals,NUM_OBJS)
    if not model.cfg.MODEL.ROI_RELATION_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
        obj_features = model.obj_feature_extractor(features, proposals, use_relu=False)
        if obj_features.ndimension() == 4:
            obj_features = torch.nn.functional.adaptive_avg_pool2d(obj_features, 1)
            obj_features = obj_features.view(obj_features.size(0), -1)
        boxes_per_image = [len(box) for box in proposals]
        obj_features = obj_features.split(boxes_per_image, dim=0)
        for proposal, obj_feature in zip(proposals, obj_features):
            proposal.add_field('box_features', obj_feature)

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

        proposal_pairs, loss_relpn = model.relation_head.relpn._relpnsample_train(proposals, targets)

    else:
        proposal_pairs = model.relation_head.relpn._relpnsample_test(proposals)

    x, obj_class_logits, pred_class_logits, obj_class_preds, rel_inds \
    = model.relation_head.rel_predictor(features, proposals, proposal_pairs) ##proposal_pair_features, obj distribution, relation_class, obj_class, relation_index

    pred_class_logits = pred_class_logits + model.relation_head.freq_bias.index_with_labels(
        torch.stack((
            obj_class_preds[rel_inds[:, 0]],
            obj_class_preds[rel_inds[:, 1]],
        ), 1)).cuda()

    if not is_train:
        proposal_pairs = model.relation_head.post_processor(pred_class_logits, proposal_pairs, x,
                                        use_freq_prior=model.relation_head.cfg.MODEL.USE_FREQ_PRIOR)
        proposals, proposal_pairs = base_model._post_processing_constrained(model,proposals, proposal_pairs)
        return (x,pred_class_logits, proposal_pairs, {})

    loss_obj_classifier = torch.tensor(0, dtype=torch.float).to(pred_class_logits.device)

    loss_pred_classifier = model.relation_head.relpn.pred_classification_loss([pred_class_logits],assign_weight = None)

    return (
        x,pred_class_logits,
        proposal_pairs,
        dict(Grcnn_OB_Cls=loss_obj_classifier,
            Grcnn_Relpn_Cls=loss_relpn,
            Grcnn_Pred_Cls=loss_pred_classifier),
    )
