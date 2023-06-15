from logging import raiseExceptions
import od_adv_box.models.Motifs as Motifs
import od_adv_box.models.Reldn as Reldn
import od_adv_box.models.Grcnn as Grcnn
import od_adv_box.models.Imp as Imp
import od_adv_box.models.Msdn as Msdn
def loss(model, input, images , targets, mode):


    pos_mode = mode[0] # Targeted
    neg_mode = mode[1] # Untargeted
    only_OD = True
    if 'Motifs' in model.name:
        module = Motifs.motifs
        # RPN_Bi  RPN_Proposal  OD_Cls   OD_Proposal
        # Motifs_OB_Cls  Motifs_Pred_Cls
    elif model.name == 'Reldn':
        module = Reldn.reldn
        # RPN_Bi   RPN_Proposal   OD_Cls  OD_Proposal   
        # Reldn_OB_Cls  Reldn_Pred_Cls   Reldn_Agnostic_Sub_Cls  Reldn_Agnostic_Obj_Cls  Reldn_Eca_Sub_Cls Reldn_Eca_Obj_Cls  Reldn_Pca_Sub_Cls  Reldn_Pca_obj_Cls
    elif model.name == 'Grcnn':
        module = Grcnn.grcnn
        # RPN_Bi   RPN_Proposal   OD_Cls  OD_Proposal   
        # Grcnn_OB_Cls  Grcnn_Relpn_Cls  Grcnn_Pred_Cls
    elif model.name == 'Imp':
        module = Imp.imp
        # Imp_OB_Cls   Imp_Pred_Cls
    elif model.name == 'Msdn':
        module = Msdn.msdn
        # Msdn_OB_Cls   Msdn_Pred_Cls
    else:
        raiseExceptions('Not Implematations')
    for mode_ in pos_mode + neg_mode:
        if model.name in mode_:
            only_OD = False
    module_result = module(model,input, images , targets, only_OD)
    if module_result is not None:
        loss_dict, proposals,proposals_pairs = module_result
    else:
        return None
    loss = 0
    for mode_ in pos_mode:
        loss +=  loss_dict[str(mode_)]
    for mode_ in neg_mode:
        loss -=  loss_dict[mode_]
    return (loss,proposals , proposals_pairs) 



def get_pred(model,GT_triplets, input, images , targets):
    is_train = False
    only_OD = False
    if 'Motifs' in model.name:
        module = Motifs.motifs
    elif model.name == 'Reldn':
        module = Reldn.reldn
    elif model.name == 'Grcnn':
        module = Grcnn.grcnn
    elif model.name == 'Imp':
        module = Imp.imp
    elif model.name == 'Msdn':
        module = Msdn.msdn
    else:
        raiseExceptions('Not Implematations')
    GT_objs = targets[0].get_field('labels').cpu().detach().numpy()
    return GT_triplets,GT_objs, module(model,input, images , targets, only_OD, is_train)

def get_loss(*args):
    return loss(*args)