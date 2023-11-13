# Copyright (c) Mn,Zhao. All Rights Reserved.
# from pyrsistent import T
import torch
from zmq import device
from maskrcnn_benchmark.structures.bounding_box import BoxList

def motifs(model,input, images , targets, only_OD =True, is_train =True):

    features, proposals, class_logits, box_regression, loss_dict_ob = object_detector(model,input, images , targets, is_train = is_train)
    
    if only_OD:
        proposals_pairs = None
        return (loss_dict_ob,proposals, proposals_pairs)
    else:
        features = [feature for feature in features]
        relation_result = relation_head(model, features, proposals, targets, is_train)
        if relation_result is None:
            return None
        x_pairs,pred_class_logits, proposals_pairs, loss_dict_rel = relation_result
        loss_dict_ob.update(loss_dict_rel)
        return (loss_dict_ob, proposals, proposals_pairs)


def object_detector(model,input, images , targets, is_train =True):
    for p in model.backbone.parameters():
        p.requires_grad = True
    for p in model.rpn.parameters():
        p.requires_grad = True
    for p in model.roi_heads.parameters():
        p.requires_grad = True

    #model.backbone.eval()
    #model.rpn.eval()
    #model.roi_heads.eval()
    if targets:
        #if model.detector_pre_calculated:
        #    targets = [target.to_cuda() for (target, prediction) in targets if target is not None]
        #else:
        targets = [target.to_cuda()#to(model.device)
                    for target in targets if target is not None]
    features = model.backbone(input)
    objectness, rpn_box_regression = model.rpn.head(features)
    anchors = model.rpn.anchor_generator(images, features)
    loss_dict = {}
    if is_train:
        proposals = model.rpn.box_selector_train(
                anchors, objectness, rpn_box_regression, targets)
        loss_objectness, loss_rpn_box_reg = model.rpn.loss_evaluator(
            anchors, objectness, rpn_box_regression, targets)
    else:
        proposals = model.rpn.box_selector_test(anchors, objectness, rpn_box_regression)
    if model.detector_force_boxes:
        proposals = [BoxList(target.bbox, target.size, target.mode) for target in targets]

    if is_train:
        proposals = model.roi_heads['box'].loss_evaluator.subsample(proposals, targets)
    x = model.roi_heads['box'].feature_extractor(features, proposals)
    class_logits, box_regression = model.roi_heads['box'].predictor(x)
    proposals = model.roi_heads['box'].post_processor((class_logits, box_regression),proposals, x)
    if is_train:
        ob_valid_labels = targets[0].get_field('ob_valid_labels')
        loss_classifier, loss_box_reg = model.roi_heads['box'].loss_evaluator(
            [class_logits], [box_regression] ,ob_valid_labels
        )
        loss_dict = dict(RPN_Bi = loss_objectness, 
        RPN_Proposal = loss_rpn_box_reg,
        OD_Cls = loss_classifier,
        OD_Proposal = loss_box_reg
        )
    return features, proposals, class_logits, box_regression,loss_dict


def get_pos_index(matched_idxs, NUM_OBJS):
        pos_idx = []

        for matched_idxs_per_image in matched_idxs:
            GT_idx_per_image = torch.nonzero(matched_idxs_per_image >= 1, as_tuple=False).squeeze(1)

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.bool
            )
            pos_idx_per_image_mask[GT_idx_per_image] = 1
            pos_idx.append(pos_idx_per_image_mask)
        return pos_idx


def get_posvalid_index(matched_idxs, NUM_OBJS):
        pos_idx = []
        for matched_idxs_per_image in matched_idxs:
            GT_idx_per_image = torch.nonzero(matched_idxs_per_image >= 1, as_tuple=False).squeeze(1)
            neg_idx_per_image = set(torch.nonzero(matched_idxs_per_image < 0, as_tuple=False).squeeze(1).cpu().detach().numpy())
            inv_pos_ids = set((GT_idx_per_image%NUM_OBJS*NUM_OBJS + GT_idx_per_image//NUM_OBJS).cpu().detach().numpy()) - set(GT_idx_per_image.cpu().detach().numpy())
            rest_ids = set(range(len(matched_idxs_per_image))) - neg_idx_per_image
            rest_ids = list(rest_ids - inv_pos_ids)
            #rest_ids = list(rest_ids)
            rest_ids = torch.tensor(rest_ids).to(GT_idx_per_image.device)

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.bool
            )
            if len(rest_ids) != 0:
                pos_idx_per_image_mask[rest_ids] = 1
            pos_idx.append(pos_idx_per_image_mask)
        return pos_idx

def subsample(model, proposals, targets):
        NUM_OBJS = len(torch.unique(targets[0].get_field("pred_labels").nonzero()))
        proposals = model.relation_head.loss_evaluator.sel_proposals(proposals,NUM_OBJS)
        # model.relation_head.cfg.MODEL.ROI_RELATION_HEAD.NEURAL_MOTIF.NUM_OBJS) # select top K based on classification scores
        labels, proposal_pairs = model.relation_head.loss_evaluator.prepare_targets(proposals, targets)# obtain relation GT labels and possible proposal pairs
        sampled_pos_inds = get_posvalid_index(labels, NUM_OBJS)# select positive samples and negtive samples (proposal pairs) 
        # sampled_pos_inds = get_pos_index(labels, NUM_OBJS)# select positive samples and negtive samples (proposal pairs)

        proposal_pairs = list(proposal_pairs)
        for labels_per_image, proposal_pairs_per_image in zip(
                labels, proposal_pairs
        ):
            proposal_pairs_per_image.add_field("labels", labels_per_image)

        for img_idx, pos_inds_img in enumerate(sampled_pos_inds):
            img_sampled_inds = torch.nonzero(pos_inds_img, as_tuple=False).squeeze(1)
            if len(img_sampled_inds) == 0:
                return None
            proposal_pairs_per_image = proposal_pairs[img_idx][img_sampled_inds]
            proposal_pairs[img_idx] = proposal_pairs_per_image
        model.relation_head.loss_evaluator._proposal_pairs = proposal_pairs
        return proposal_pairs
        

def relation_head(model, features, proposals, targets,is_train):

        # assign_weight： assign_weight = 0 only positive proposal pairs participate in the calculation 
        # assign_weight = 1 all proposal pairs share the same constraint weight 
        if is_train:
            proposal_pairs = subsample(model, proposals, targets)
        else:
            proposal_pairs = model.relation_head._get_proposal_pairs(proposals)
        #print(pred_class_logits.shape)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        if proposal_pairs is None:
            return None
        if proposal_pairs[0].bbox.shape[0] == 0:
            return None
        x, obj_class_logits, pred_class_logits, obj_class_preds, rel_inds\
            = model.relation_head.rel_predictor(features, proposals, proposal_pairs) ##proposal_pair_features, obj distribution, relation_class, obj_class, relation_index
        if model.relation_head.use_bias:
            pred_class_logits = pred_class_logits + model.relation_head.freq_bias.index_with_labels(
                torch.stack((
                    obj_class_preds[rel_inds[:, 0]],
                    obj_class_preds[rel_inds[:, 1]],
                ), 1)).cuda()
            
        if not is_train:
            proposal_pairs = model.relation_head.post_processor(pred_class_logits, proposal_pairs, x,
                                         use_freq_prior=model.relation_head.cfg.MODEL.USE_FREQ_PRIOR)
            if model.cfg.MODEL.ROI_RELATION_HEAD.POSTPROCESS_METHOD == 'constrained':  
                # constrained just select one potential predicate for each triplet (return the position -- index), 
                # however, unconstrained return two possible predicate for each triplet, (return (x, y), pos is denoted as x*[dimension of x axis] + y)
                proposals, proposal_pairs = _post_processing_constrained(model,proposals, proposal_pairs)
            return (x,pred_class_logits, proposal_pairs, {})

        loss_obj_classifier = model.relation_head.loss_evaluator.obj_classification_loss(proposals, [obj_class_logits])
        loss_pred_classifier = model.relation_head.loss_evaluator([pred_class_logits], assign_weight = 0)
        return (
            x, pred_class_logits,
            proposal_pairs,
            dict(
            Motifs_OB_Cls = loss_obj_classifier,
            Motifs_Pred_Cls = loss_pred_classifier)
        )

def _post_processing_constrained(model,result_obj, result_pred, TRIPLETS_PER_IMG = 15):
    """
    Arguments:
        object_predictions, predicate_predictions

    Returns:
        sort the object-predicate triplets, and output the top
    """
    result_obj_new, result_pred_new = [], []
    assert len(result_obj) == len(result_pred), "object list must have equal number to predicate list"
    for result_obj_i, result_pred_i in zip(result_obj, result_pred):
        obj_scores = result_obj_i.get_field("scores")
        rel_inds = result_pred_i.get_field("idx_pairs")
        pred_scores = result_pred_i.get_field("scores")
        scores = torch.stack((
            obj_scores[rel_inds[:,0]],
            obj_scores[rel_inds[:,1]],
            pred_scores[:,1:].max(1)[0]
        ), 1).prod(1)
        scores_sorted, order = scores.sort(0, descending=True)

        result_pred_i = result_pred_i[order[: TRIPLETS_PER_IMG]]
        result_obj_new.append(result_obj_i)
        result_pred_i.add_field('labels', result_pred_i.get_field("scores")[:, 1:].argmax(dim=1) + 1) # not include background
        result_pred_i.add_field('scores_all', result_pred_i.get_field('scores'))
        result_pred_i.add_field('scores', scores[order[:TRIPLETS_PER_IMG]])
        # filter out bad prediction
        inds = result_pred_i.get_field('scores') > model.cfg.MODEL.ROI_RELATION_HEAD.POSTPROCESS_SCORE_THRESH
        result_pred_i = result_pred_i[inds]

        result_pred_new.append(result_pred_i)
    return result_obj_new, result_pred_new
