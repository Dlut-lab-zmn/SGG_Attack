from dis import dis
from logging import raiseExceptions
import torch
import numpy as np
import random
import cv2
import os
from itertools import combinations,permutations

def modify_target_force_box(current_targets, CONSTANT_TARGETS):
    possible_labels,possible_ind,ConsT_tree_nodes = get_possible_labels(current_targets, CONSTANT_TARGETS)
    find_flag = False
    if len(possible_labels) > 0:
        FINAL_DIST = 10000
        find_flag = True
        rel_ = CONSTANT_TARGETS[0].get_field('pred_labels').nonzero()
        for i in range(len(possible_labels)):
            new_rel_ = torch.zeros_like(rel_)-1
            for j, ind in enumerate(ConsT_tree_nodes):
                new_rel_[rel_== ind] = possible_ind[i][j]
            valid_ind = (new_rel_[:,0]>=0) * (new_rel_[:,1]>=0)
            new_rel_ = new_rel_[valid_ind]
            cloned_rel_ = rel_.clone()[valid_ind]
            dist = get_dis((current_targets, CONSTANT_TARGETS), (new_rel_, cloned_rel_))

            if dist < FINAL_DIST :
                FINAL_DIST = dist
                Final_possible = possible_labels[i]
                Final_CT_pred_labels = torch.zeros_like(current_targets[0].get_field('pred_labels'))
                CONSTANT_PRED = CONSTANT_TARGETS[0].get_field('pred_labels')[cloned_rel_[:,0],cloned_rel_[:,1]]
                Final_CT_pred_labels[new_rel_[:,0],new_rel_[:,1]] = CONSTANT_PRED
        
        
        
        remain_objs = list(set(np.array(CONSTANT_TARGETS[0].get_field('labels').squeeze())) - \
                                                                                                                                                        set(np.array(CONSTANT_TARGETS[0].get_field('labels')[rel_.reshape(-1)])))
        ro_ls = []
        for remain_obj in remain_objs:
            ro_ls.append(np.argmax(np.array(CONSTANT_TARGETS[0].get_field('labels').squeeze()) == remain_obj))
        target_bbox = np.array(CONSTANT_TARGETS[0].bbox[ro_ls])
        
        source_objs = np.array(((current_targets[0].get_field('labels') - Final_possible)==0).nonzero().squeeze(1))
        min_len = min(len(source_objs), len(ro_ls))
        target_bbox = target_bbox[:min_len]
        source_bbox = np.array(current_targets[0].bbox)

        if min_len>0:
            best_ = bbox_dist(source_bbox, target_bbox, source_objs, min_len)
            Final_possible[best_] = CONSTANT_TARGETS[0].get_field('labels')[ro_ls[:min_len]]
        current_targets[0].add_field('labels', Final_possible)
        current_targets[0].add_field('pred_labels', Final_CT_pred_labels)
    return find_flag

def bbox_dist(source_bbox, target_bbox, source_objs, min_len):
    #print(source_objs)
    #print(min_len)
    all_possible = np.array(list(permutations(source_objs,min_len)))
    source_bbox = np.clip(source_bbox, 0,10000)
    target_bbox = np.clip(target_bbox, 0,10000)
    max_dist = 10000

    for i in range(len(all_possible)):
        source_width = source_bbox[all_possible[i]][:,2] - source_bbox[all_possible[i]][:,0]
        source_high = source_bbox[all_possible[i]][:,3] - source_bbox[all_possible[i]][:,1]
        target_width = target_bbox[:,3] - target_bbox[:,1]
        target_high = target_bbox[:,3] - target_bbox[:,1]
        bbox_dist = np.mean((source_high*source_width)/(target_high*target_width) +  (source_high/source_width)*(target_high/target_width))
        #print(np.mean((source_high*source_width)/(target_high*target_width)), np.mean((source_high/source_width)*(target_high/target_width)))
        if bbox_dist < max_dist:
            max_dist = bbox_dist
            best_ = all_possible[i]
    return best_
def get_possible_labels(current_targets, CONSTANT_TARGETS):

    CT_box = current_targets[0].bbox
    CT_labels = current_targets[0].get_field('labels')

    # find all intersections
    CT_matrix = torch.zeros((len(CT_box), len(CT_box)))
    for line in range(CT_matrix.size()[0]):
        line_box = CT_box[line]
        for column in range(CT_matrix.size()[1]):
            if line < column:
                column_box = CT_box[column]
                flag1 = (line_box[0] < column_box[2]) &( line_box[1] < column_box[3]) &(column_box[2] < line_box[2]) & (column_box[3] < line_box[3])
                flag2 = (column_box[0] < line_box[2]) & (column_box[1] < line_box[3])&(line_box[2] < column_box[2]) & (line_box[3] < column_box[3])
                if flag1 or flag2:
                    CT_matrix[line, column] = 1
    # resort source labels by degree
    node_degree, node_sorted_index = torch.sort(torch.sum(CT_matrix>0,1) + torch.sum(CT_matrix>0,0),descending=True)
    CT_sorted_labels = CT_labels[node_sorted_index]

    # find all intersections by resorted labels
    node_map = torch.zeros_like(node_sorted_index)
    node_map[node_sorted_index] = torch.arange(len(node_sorted_index))
    new_node_poses = node_map[CT_matrix.nonzero()]
    for i in range(len(new_node_poses)):
        new_node_pose = new_node_poses[i].clone()
        if new_node_pose[0] > new_node_pose[1]:
            new_node_poses[i,0] = new_node_pose[1]
            new_node_poses[i,1] = new_node_pose[0]
    new_CT_matrix = torch.zeros_like(CT_matrix)
    new_CT_matrix[new_node_poses[:,0],new_node_poses[:,1]] = 1
    cloned_CT_matrix = new_CT_matrix.clone()


    # find all intersections by resorted labels
    ConsT_matrix, ConsT_sorted_labels, ConsT_node_index = confuse_matrix(CONSTANT_TARGETS)
    cloned_ConsT_matrix = ConsT_matrix.clone()

    ConsT_tree = {}
    ConsT_labels = CONSTANT_TARGETS[0].get_field('labels')
    #ConsT_tree = deep_tree(ConsT_tree, torch.arange(len(ConsT_labels)), cloned_ConsT_matrix,  2, True)
    ConsT_tree, ConsT_subtree= DFSFD(cloned_ConsT_matrix)
    CT_tree = {}
    #CT_tree = deep_tree(CT_tree, torch.arange(len(CT_labels)),cloned_CT_matrix, 2,  True)
    CT_tree, CT_subtree = DFSFD(cloned_CT_matrix, len(ConsT_tree['0']))
    print(ConsT_subtree)


    for CT_node_key in list(CT_tree.keys()):
        CT_tree[CT_node_key] = list(CT_tree[CT_node_key].numpy())
    for ConsT_node_key in list(ConsT_tree.keys()):
        ConsT_tree[ConsT_node_key] = list(ConsT_tree[ConsT_node_key].numpy())
    d1 = ConsT_tree
    d2 = CT_tree

    ConsT_tree_nodes = []
    for ConsT_tree_value in ConsT_tree.values():
        ConsT_tree_nodes += ConsT_tree_value
    ConsT_tree_num = len(ConsT_tree_nodes)
    final_all_pos_L = []
    possible_ind = []
    # if len(d1['base']) == 1:
    # for L_1_d1 in d1['base']:
    L_2_d1 = np.array(d1.get('0',[]))
    deep_L_2_d1 = len(L_2_d1)
    L_2_d2 = np.array(d2.get('0',[]))
    deep_L_2_d2 = len(L_2_d2)
    if deep_L_2_d2 >= deep_L_2_d1:
        # min(len(L_2_d2), deep_L_2_d1*2)
        sel_all_pos_L_2_in_d2_for_d1 = np.array(list(permutations(np.arange(min(len(L_2_d2), deep_L_2_d1*2)),deep_L_2_d1)))
        all_pos_L_2_in_d2_for_d1 = L_2_d2[sel_all_pos_L_2_in_d2_for_d1]

        for sel_L_2_ind, pos_L_2_in_d2_for_d1_ind in enumerate(all_pos_L_2_in_d2_for_d1):
            sel_pos_L2_ind = sel_all_pos_L_2_in_d2_for_d1[sel_L_2_ind]
            all_pos_L_3_in_d2_for_d1_ind = {}
            flag = True
            for L_3_d1_ind in range(len(L_2_d1)):
                L_3_d1 = d1.get( '0_' + str(L_3_d1_ind),[])
                if len(L_3_d1) >0:
                    L_3_d2 = d2.get('0_' + str(sel_pos_L2_ind[L_3_d1_ind]),[])
                    if len(L_3_d2) >= len(L_3_d1):
                        pos_L_3_in_d2_for_d1_ind =torch.tensor( list(permutations(L_3_d2,len(L_3_d1)))).view(-1,len(L_3_d1)).numpy()
                        all_pos_L_3_in_d2_for_d1_ind[str(L_3_d1_ind)] =pos_L_3_in_d2_for_d1_ind
                    else:
                        flag = False
                        break
            if flag and len(all_pos_L_3_in_d2_for_d1_ind)>0:
                pos_L2_L1 =np.concatenate((torch.tensor(0).view(-1).numpy() ,  torch.tensor(pos_L_2_in_d2_for_d1_ind).view(-1).numpy()  ))
                start_nodes = list(all_pos_L_3_in_d2_for_d1_ind.values())[0]
                if len(start_nodes.shape) == 1:
                    start_nodes = np.expand_dims(start_nodes, 1)
                for i in range(len(all_pos_L_3_in_d2_for_d1_ind)-1):
                    node_sum_1 = len(start_nodes)
                    cur_node = list(all_pos_L_3_in_d2_for_d1_ind.values())[i + 1]
                    node_sum_2 = len(cur_node)
                    node_1 = np.tile(np.array(start_nodes),(node_sum_2,1))
                    node_2 = np.array(cur_node).repeat(node_sum_1)
                    if len(node_2.shape) == 1:
                        node_2 = np.expand_dims(node_2, 1)
                    start_nodes = np.concatenate((node_1,node_2),1)
                pos_L2_L1 = np.repeat(np.expand_dims(pos_L2_L1,0), start_nodes.shape[0],0)
                pos_L = np.concatenate((pos_L2_L1,start_nodes), 1)
                for i in range(len(pos_L)):

                    if len(np.unique(pos_L[i])) == len(pos_L[i]):
                        possible_ind.append(node_sorted_index[pos_L[i]])
                        new_sorted_CT_labels = CT_sorted_labels.clone()


                        new_sorted_CT_labels[pos_L[i]] = ConsT_sorted_labels[ConsT_tree_nodes]
                        new_CT_labels = torch.zeros_like(CT_labels)
                        new_CT_labels[node_sorted_index] = new_sorted_CT_labels

                        final_all_pos_L.append( new_CT_labels )
            elif flag and len(all_pos_L_3_in_d2_for_d1_ind)==0:
                pos_L =np.concatenate((torch.tensor(0).view(-1).numpy() ,  torch.tensor(pos_L_2_in_d2_for_d1_ind).view(-1).numpy()  ))
                if len(np.unique(pos_L)) == len(pos_L):
                    possible_ind.append(node_sorted_index[pos_L])

                    new_sorted_CT_labels = CT_sorted_labels.clone()

                    new_sorted_CT_labels[pos_L] = ConsT_sorted_labels[ConsT_tree_nodes]
                    new_CT_labels = torch.zeros_like(CT_labels)
                    new_CT_labels[node_sorted_index] = new_sorted_CT_labels
                    final_all_pos_L.append( new_CT_labels )
    #print(possible_ind)
    #print(ConsT_node_index[ConsT_tree_nodes])
    return final_all_pos_L, possible_ind, ConsT_node_index[ConsT_tree_nodes]

dict_triplet = {}
import random
def construct_graph(degree):

    dict_triplet = np.load('triplet.npy',allow_pickle=True).item()
    dict_bbox = np.load('bbox.npy',allow_pickle=True).item()
    base_node = random.randint(1, len(dict_triplet.keys()))
    max_L1_triplet_num = len(dict_triplet[base_node])
    if len(np.unique(np.array(dict_triplet[base_node])[:,2])) > degree[1]:
        L1_node_list = [base_node]
        L1_rel_list = []
        L1_box_list = []
        while len(L1_node_list)<(degree[0] + degree[1]):
            L1_o_node_ind = random.randint(1, max_L1_triplet_num-1)
            L1_o_node = dict_triplet[base_node][L1_o_node_ind][2]
            L1_s_o_rel = dict_triplet[base_node][L1_o_node_ind]
            L1_s_o_box= dict_bbox[base_node][L1_o_node_ind]
            if L1_o_node not in L1_node_list:
                L1_node_list.append(L1_o_node)
                L1_rel_list.append(L1_s_o_rel)
                L1_box_list.append(L1_s_o_box)
        for L2_s_ind, L2_s_node in enumerate(L1_node_list[degree[0]:(degree[0] + degree[1])]):
            max_L2_triplet_num = len(dict_triplet[L2_s_node])
            L2_node_list = []
            L2_rel_list = []
            L2_box_list = []
            if len(np.unique(np.array(dict_triplet[L2_s_node])[:,2])) > degree[2+L2_s_ind]:
                while len(L2_node_list) < degree[2+L2_s_ind]:
                    L2_o_node_ind = random.randint(1, max_L2_triplet_num)
                    L2_o_node = dict_triplet[L2_s_node][L2_o_node_ind][2]
                    L2_s_o_rel = dict_triplet[L2_s_node][L2_o_node_ind]
                    L2_s_o_box = dict_bbox[L2_s_node][L2_o_node_ind]
                    if L2_o_node not in L2_node_list:
                        L2_node_list.append(L2_o_node)
                        L2_rel_list.append(L2_s_o_rel)
                        L2_box_list.append(L2_s_o_box)
                L1_rel_list += L2_rel_list
                L1_box_list += L2_box_list
            else:
                construct_graph(degree)
    else:
        construct_graph(degree)
    return L1_rel_list, L1_box_list


def iou(targets, rel_):
    sub_rel_box = targets[0].bbox[rel_][:,0]
    obj_rel_box = targets[0].bbox[rel_][:,1]
    sub_center = ((sub_rel_box[:, 3] + sub_rel_box[:, 1])/2, (sub_rel_box[:, 2] + sub_rel_box[:, 0])/2)
    obj_center = ((obj_rel_box[:, 3] + obj_rel_box[:, 1])/2, (obj_rel_box[:, 2] + obj_rel_box[:, 0])/2)
    sub_rel_area = (sub_rel_box[:, 3] - sub_rel_box[:, 1]) * (sub_rel_box[:, 2] - sub_rel_box[:, 0])
    obj_rel_area = (obj_rel_box[:, 3] - obj_rel_box[:, 1]) * (obj_rel_box[:, 2] - obj_rel_box[:, 0])
    lt_rel = torch.max(sub_rel_box[:, :2], obj_rel_box[:, :2]) 
    rb_rel = torch.min(sub_rel_box[:, 2:], obj_rel_box[:, 2:]) 
    wh_rel = (rb_rel - lt_rel).clamp(min=0)  # [N,M,2]
    inter_rel = wh_rel[:, 0] * wh_rel[:, 1]
    iou_rel = inter_rel / torch.min(sub_rel_area,obj_rel_area)

    iou_rel2 = inter_rel / obj_rel_area 
    iou= inter_rel / (sub_rel_area+obj_rel_area -inter_rel)
    return iou_rel, iou , sub_rel_area, obj_rel_area, sub_center,obj_center

def get_dis(targets_tuple, rel_tuple):
    iou_rel_0, iou_0,  sub_rel_area_0, obj_rel_area_0, sub_center_0,obj_center_0 = iou(targets_tuple[0], rel_tuple[0])
    iou_rel_1, iou_1,  sub_rel_area_1, obj_rel_area_1, sub_center_1,obj_center_1 = iou(targets_tuple[1], rel_tuple[1])
    angle_0 = (obj_center_0[0] - sub_center_0[0])/(obj_center_0[1] - sub_center_0[1] + 1)
    angle_1 = (obj_center_1[0] - sub_center_1[0])/(obj_center_1[1] - sub_center_1[1] + 1)
    angle_diff = angle_1 - angle_0
    sub_rel_diff = torch.abs(sub_rel_area_1 - sub_rel_area_0)/ (sub_rel_area_1 + sub_rel_area_0)
    obj_rel_diff = torch.abs(obj_rel_area_1 - obj_rel_area_0)/ (obj_rel_area_1 + obj_rel_area_0)
    iou_rel_diff = torch.abs(iou_rel_1 - iou_rel_0)
    iou_diff = torch.abs(iou_0 - iou_1)

    diff =  torch.sum((sub_rel_diff + obj_rel_diff)/2.)## torch.sum(angle_diff)#
    return diff
def id_to_name(cfg):
    import yaml
    import json
    yaml_path = os.path.join('./datasets', cfg.DATASETS.TRAIN[0])
    dir_yaml_path = os.path.dirname(yaml_path)
    with open(yaml_path, 'r', encoding='utf-8') as f: 
        labelmap = yaml.load(f.read(), Loader=yaml.FullLoader)['labelmap']
        with open(os.path.join(dir_yaml_path, labelmap), "r") as f:
            row_data = json.load(f)
            id_to_label =row_data['idx_to_label']
            idx_to_predicate =  row_data['idx_to_predicate']
    return id_to_label, idx_to_predicate
        # config = yaml.load(f.read(), Loader=yaml.FullLoader) 
        # print(config)
        # id_to_name = config['labelmap']
def dipict_obj_img(img, targets, id_to_label):
        cpu_target_box = targets[0].bbox.cpu().detach().numpy()
        cpu_target_label = targets[0].get_field("labels").cpu().detach().numpy()
        cpu_target_confi =  targets[0].get_field("confis").cpu().detach().numpy()
        for k in range(cpu_target_box.shape[0]):
            bbox = cpu_target_box[k]

            gt_label = cpu_target_label[k]
            gt_label_name = id_to_label[str(gt_label)]
            gt_confi = cpu_target_confi[k]
            pixel1 = (int(bbox[0]), int(bbox[1]))
            pixel2 = (int(bbox[2]), int(bbox[3]))
            color = tuple([random.randint(0,255) for _ in range(3)])
            cv2.rectangle(img, pixel1, pixel2, color, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            img = cv2.putText(img, '{} {:.3f}'.format(gt_label_name,gt_confi), (int(bbox[0]-25), int(bbox[1]+15)), font, 0.7,color, 2)
        return img

def dipict_pred_img(img, targets, id_to_label, idx_to_predicate, flag = 'GT'):

        id_to_label['0'] = None
        cpu_target_box = targets[0].bbox.cpu().detach().numpy()
        cpu_target_label = targets[0].get_field("labels").cpu().detach().numpy()
        points = []
        if flag =='GT':
            cpu_pred_label = targets[0].get_field("pred_labels").cpu().detach().numpy()
            cpu_pred_index = cpu_pred_label.nonzero()
            start_pred_index = cpu_pred_index[0]
            end_pred_index = cpu_pred_index[1]
        else:
            cpu_pred_label = targets[0].get_field("pred_labels").cpu().detach().numpy()
            cpu_pred_index = targets[0].get_field("idx_pairs").cpu().detach().numpy()
            start_pred_index = cpu_pred_index[:, 0]
            end_pred_index = cpu_pred_index[:, 1]
        gt_label_names = []
        for k in range(cpu_target_box.shape[0]):
            bbox = cpu_target_box[k]
            gt_label = cpu_target_label[k]
            gt_label_name = id_to_label[str(gt_label)]
            gt_label_names.append(gt_label_name)
            pixel1 = (int(bbox[0]), int(bbox[1]))
            pixel2 = (int(bbox[2]), int(bbox[3]))
            color = tuple([random.randint(0,255) for _ in range(3)])
            point = (int((bbox[0] + bbox[2])/2), int((bbox[1] + bbox[3])/2))
            points.append(point)
            if k in start_pred_index or k in end_pred_index:
                    cv2.rectangle(img, pixel1, pixel2, color, 2) # (255, 0, 0)
                    cv2.circle(img,point , 3, color, 3) # 
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    img = cv2.putText(img, '{}'.format(gt_label_name), (int(bbox[0]), int(bbox[1]+15)), font, 0.7,color, 2)
        triplets = []
        for k in range(start_pred_index.shape[0]):
            start_point_index = start_pred_index[k]
            end_point_index = end_pred_index[k]
            if flag =='GT':
                rel = cpu_pred_label[start_point_index, end_point_index]
            else:
                rel = cpu_pred_label[k]
            predicate = idx_to_predicate[str(rel)]
            start_point = points[start_point_index]
            end_point = points[end_point_index]
            dist = np.sqrt(np.square(start_point[0] - end_point[0]) + np.square(start_point[1] - end_point[1]))
            triplet = (gt_label_names[start_point_index], predicate, gt_label_names[end_point_index])
            if triplet not in triplets:
                triplets.append(triplet)
            cv2.arrowedLine(img, start_point, end_point, (0, 0, 255), 2,tipLength = 15/dist)
            img = cv2.putText(img, '{}'.format(predicate), (int((start_point[0] + end_point[0])/2), int((start_point[1] + end_point[1])/2)), font, 0.7,(0, 255, 0), 2)
        print(triplets)
        return img
def dipict_pred_img2(img, targets, id_to_label, idx_to_predicate, flag = 'GT'):

        id_to_label['0'] = None
        cpu_target_box = targets[0].bbox.cpu().detach().numpy()
        cpu_target_label = targets[0].get_field("labels").cpu().detach().numpy()
        points = []
        if flag =='GT':
            cpu_pred_label = targets[0].get_field("pred_labels").cpu().detach().numpy()
            cpu_pred_index = cpu_pred_label.nonzero()
            start_pred_index = cpu_pred_index[0]
            end_pred_index = cpu_pred_index[1]
        else:
            cpu_pred_label = targets[0].get_field("pred_labels").cpu().detach().numpy()
            cpu_pred_index = targets[0].get_field("idx_pairs").cpu().detach().numpy()
            start_pred_index = cpu_pred_index[:, 0]
            end_pred_index = cpu_pred_index[:, 1]
        gt_label_names = []
        for k in range(cpu_target_box.shape[0]):
            bbox = cpu_target_box[k]
            gt_label = cpu_target_label[k]
            gt_label_name = id_to_label[str(gt_label)]
            gt_label_names.append(gt_label_name)
            pixel1 = (int(bbox[0]), int(bbox[1]))
            pixel2 = (int(bbox[2]), int(bbox[3]))
            color = tuple([random.randint(0,255) for _ in range(3)])
            point = (int((bbox[0] + bbox[2])/2), int((bbox[1] + bbox[3])/2))
            points.append(point)
            if str(gt_label) != '0':
                cv2.rectangle(img, pixel1, pixel2, color, 2) # (255, 0, 0)
                cv2.circle(img,point , 3, color, 3) # 
                font = cv2.FONT_HERSHEY_SIMPLEX
                img = cv2.putText(img, '{}'.format(gt_label_name), (int(bbox[0]), int(bbox[1]+15)), font, 0.7,color, 2)
        triplets = []
        for k in range(start_pred_index.shape[0]):
            start_point_index = start_pred_index[k]
            end_point_index = end_pred_index[k]
            if flag =='GT':
                rel = cpu_pred_label[start_point_index, end_point_index]
            else:
                rel = cpu_pred_label[k]
            predicate = idx_to_predicate[str(rel)]
            start_point = points[start_point_index]
            end_point = points[end_point_index]
            dist = np.sqrt(np.square(start_point[0] - end_point[0]) + np.square(start_point[1] - end_point[1]))
            triplet = (gt_label_names[start_point_index], predicate, gt_label_names[end_point_index])
            if triplet not in triplets:
                triplets.append(triplet)
            cv2.arrowedLine(img, start_point, end_point, (0, 0, 255), 2,tipLength = 15/dist)
            img = cv2.putText(img, '{}'.format(predicate), (int((start_point[0] + end_point[0])/2), int((start_point[1] + end_point[1])/2)), font, 0.7,(0, 255, 0), 2)
        print(triplets)
        return img
def confuse_matrix(targets):
        node_labels = targets[0].get_field('labels')
        matrix_label = targets[0].get_field('pred_labels')

        node_degree, node_index = torch.sort(torch.sum(matrix_label>0,1) + torch.sum(matrix_label>0,0),descending=True)

        node_map = torch.zeros_like(node_labels)
        node_map[node_index] = torch.arange(len(node_index))

        new_node_poses = node_map[matrix_label.nonzero()]

        for i in range(len(new_node_poses)):
            new_node_pose = new_node_poses[i].clone()

            if new_node_pose[0] > new_node_pose[1]:
                new_node_poses[i,0] = new_node_pose[1]
                new_node_poses[i,1] = new_node_pose[0]
        matrix = torch.zeros_like(matrix_label)
        matrix[new_node_poses[:,0],new_node_poses[:,1]] = 1
        # matrix[1,5] = 1
        # matrix[2,6] = 1
        # matrix[3,5] = 1
        return matrix, node_labels[node_index],node_index
def found_branch(matrix, node_):
    line_branch = matrix[:,node_].squeeze().nonzero()
    column_branch = matrix[node_,:].squeeze().nonzero()
    branch_node = torch.cat((line_branch, column_branch),0)#.squeeze()
    branch_node = branch_node.view(branch_node.shape[0])
    matrix[line_branch, node_] = 0
    matrix[node_, column_branch] = 0
    return branch_node, matrix
def deep_tree(tree, node_sorted_index, matrix, assign_deep = -1, only_base = False):
    max_deep = 0
    tree['base'] = torch.tensor([], dtype= torch.uint8)
    for search_base_step, search_node in enumerate(node_sorted_index):
        if only_base:
            if search_node>=1:
                    return tree
        base_branch_nodes, matrix =found_branch(matrix, search_node)
        if len(base_branch_nodes) > 0:
            tree['base'] = torch.cat((tree['base'], search_node.view(-1)))
            tree[str(search_base_step)] = base_branch_nodes
            if max_deep <= 1:
                max_deep = 1
            if assign_deep == -1 or assign_deep >=2 :
                for search_l2_step, search_l2_node in enumerate(base_branch_nodes):
                    l2_branch_nodes, matrix =found_branch(matrix, search_l2_node)
                    if len(l2_branch_nodes) > 0:

                        tree[str(search_base_step) + '_' + str(search_l2_step)] = l2_branch_nodes
                        if max_deep <= 2:
                            max_deep = 2
                        if assign_deep == -1 or assign_deep >=3 :
                            for search_l3_step, search_l3_node in enumerate(l2_branch_nodes):
                                l3_branch_nodes, matrix =found_branch(matrix, search_l3_node)
                                if len(l3_branch_nodes) > 0:
                                    tree[str(search_base_step) + '_' + str(search_l2_step)+ '_' + str(search_l3_step)] = l3_branch_nodes
                                    if max_deep <= 3:
                                        max_deep = 3
                                    if assign_deep == -1 or assign_deep >=4 :
                                        for search_l4_step, search_l4_node in enumerate(l3_branch_nodes):
                                            l4_branch_nodes, matrix =found_branch(matrix, search_l4_node)
                                            if len(l4_branch_nodes) > 0:
                                                if max_deep <= 4:
                                                    max_deep = 4
                                                tree[str(search_base_step) + '_' + str(search_l2_step)+ '_' + str(search_l3_step)+ '_' + str(search_l4_step)] = l4_branch_nodes
    return tree


def DFSFD(m1,min_len = None):
    dic = {'base':torch.tensor([0])}
    m1_line1 = torch.zeros_like(m1[0])
    m1_line1[0] = 1
    m1_line1[1:] = m1[0,1:] 
    remove_line_inds = (m1_line1 == 0).nonzero()
    retain_line_inds = (m1_line1 == 1).nonzero()

    removed_lines = m1[remove_line_inds]

    potential_II_nodes = list(np.array(retain_line_inds[1:].squeeze(1)))
    
    rel_II_nodes = set(potential_II_nodes.copy())

    for II_node in potential_II_nodes:
        base_line = m1[0] 
        if II_node in rel_II_nodes:
            second_line = m1[II_node]
            fake_II_nodes = (((second_line - base_line) == 0)  *  (base_line == 1)).nonzero().squeeze(1)
            if len(fake_II_nodes) >0:
                rel_II_nodes  = rel_II_nodes - set(np.array(fake_II_nodes))
    rel_II_nodes = list(rel_II_nodes)
    if min_len is not None:
        min_len = min(min_len,len(potential_II_nodes))
        if len(rel_II_nodes) < min_len:
            remain_II_nodes = list(set(potential_II_nodes) - set(rel_II_nodes))
            rel_II_nodes = rel_II_nodes + remain_II_nodes[-(min_len - len(rel_II_nodes) ):]
    rel_II_nodes =  np.array(rel_II_nodes)
    rel_II_nodes.sort(0)
    modified_m1 = torch.zeros_like(m1)
    modified_m1[0, rel_II_nodes] = 1
    modified_m1[rel_II_nodes] = m1[rel_II_nodes].detach().clone()

    
    for ind, II_node in enumerate(rel_II_nodes):
        II_line_sum = m1[0]
        for i in range(ind+1):
            II_line_sum += modified_m1[rel_II_nodes[i]]
        remove_nodes = (II_line_sum > 1).nonzero()
        modified_m1[II_node][remove_nodes] = 0
    dic['0'] = torch.tensor(rel_II_nodes)
    for II_node_ind, II_node in enumerate(rel_II_nodes):
        str_name = '0_' + str(II_node_ind)
        III_nodes = torch.tensor(modified_m1[II_node]).nonzero().squeeze(1)
        if len(III_nodes)>0:
            dic[str_name] = III_nodes
    dic2 = {}
    all_main_tree_nodes = torch.sum(modified_m1,0)
    all_main_tree_nodes[0] = 1
    remove_line_inds = np.array((m1_line1 == 0).nonzero().squeeze(1))
    remain_lines = m1[remove_line_inds] * (1 - all_main_tree_nodes)

    for i in range(len(remain_lines)):
        if len(remain_lines[i].nonzero().squeeze(1)) > 0:
            dic2[str(remove_line_inds[i])] = remain_lines[i].nonzero().squeeze(1)

    return dic,dic2
