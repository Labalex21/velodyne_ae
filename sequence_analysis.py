# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 16:38:14 2017

@author: schlichting
"""

import numpy as np
import json
import os.path
from sklearn.cluster import KMeans
import time
import pickle
from sklearn.neighbors import KDTree

def import_encoder_traj(filename):
    f = open(filename,'r')
    data = json.load(f)
    values = np.array(data["encoder"])
    traj = np.array(data["trajectory"])
    
    return values, traj
    
def getDist2(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
    
def getLabelMatrix(labels, number_of_features):
    m = np.zeros((labels.shape[0]-number_of_features,number_of_features))
    for i in range(m.shape[0]):
        startIdx = i
        endIdx = i+number_of_features
        m[i,:] = labels[startIdx:endIdx]
    return m
    
def getMinIdx(seq, matrix):
    m_bool = np.array(matrix == seq)
    v_bool = np.array((m_bool == 1).sum(1))
    idxMax = np.argmax(v_bool)+seq.shape[0], np.max(v_bool)
    #while checkMax(idxMax[0]) == False:
    #    v_bool[idxMax[0]-seq.shape[0]] = 0
    #    idxMax = np.argmax(v_bool)+seq.shape[0], np.max(v_bool)
    return idxMax
    
def filter_by_tree(tree, traj, features):
    distances = tree.query_radius(traj[:,0:2],r = 10,count_only=True)
    traj = traj[(distances == 0)]
    features = features[(distances == 0)]
    return features, traj

def interpolate_labels(traj, labels, label_dist = 0.3):
    new_traj = []
    new_labels = []
    last_point = traj[0]
    last_label = labels[0]
    for i in range(1,traj.shape[0]):
        point_dist = np.linalg.norm(traj[i]-last_point)
        if point_dist < label_dist:
            continue
        if point_dist > 50:
            last_point = traj[i]
            last_label = labels[i]
            continue
        number_points = int(point_dist/label_dist)
        for j in range(number_points):
            w1 = (1-(j+1)*(label_dist)/point_dist)
            w2 = (j+1)*(label_dist)/point_dist
            new_pos = w1*last_point + w2*traj[i]
            new_traj.append(new_pos)

            if w1 > w2:
                new_labels.append(last_label)
            else:
                new_labels.append(labels[i])
        last_point = new_pos
        last_label = labels[i]
    return np.array(new_traj), np.array(new_labels)


def get_results(path_online, path_ref_vector, cluster_size, sequence_length):
    print("load data")
    
    reference, trajectory_r = import_encoder_traj(path_ref_vector[0])
    #tree_ref = KDTree(trajectory_r[:,0:2])
    for i in range(1,len(path_ref_vector)):
        reference_tmp, trajectory_r_tmp = import_encoder_traj(path_ref_vector[i])
        #print(reference_tmp.shape)
        #reference_tmp, trajectory_r_tmp = filter_by_tree(tree_ref, trajectory_r_tmp,reference_tmp)
        #print(reference_tmp.shape)
        reference = np.concatenate((reference,reference_tmp), axis=0)
        trajectory_r = np.concatenate((trajectory_r,trajectory_r_tmp), axis=0)
    features, trajectory_f = import_encoder_traj(path_online)
    
    print("sample data")
    trajectory_r, reference  = interpolate_labels(trajectory_r,reference,0.3)
    trajectory_f, features = interpolate_labels(trajectory_f,features,0.3)
    
    print("kmeans")
    kmeans = KMeans(n_clusters=cluster_size).fit(reference)    
    labels = kmeans.labels_
    
    print("kd-tree")
    tree = KDTree(trajectory_r[:,0:2])
    traj_distances = tree.query_radius(trajectory_f[:,0:2],r = 5.,count_only=True)

     # get label matrix for sequence analysis
    mLabels = getLabelMatrix(labels,sequence_length)
                   
    #check_points = np.array(features).shape[0]-sequence_length # All points
    check_points = 5000
            
    distances = np.zeros(check_points)
    scores = np.zeros(check_points)
    indices = np.zeros(check_points)
    refPositions = np.zeros([check_points,2])
    maxPositions = np.zeros([check_points,3])
    
    print("sequence analysis")
    delete_rows = 0
    for i in range(0,check_points):
        end = int(features.shape[0]/check_points)*i
        start = end-sequence_length
        
        if i%500 == 0 and i != 0:
            print(i,"/",check_points)
        
        # check if last n (sequence length) positions are available
        if start < 0:
            delete_rows += 1
            continue
        
        posRef = trajectory_f[end] # Real position
        refPositions[i] = posRef[0:2]
        indices[i] = end # Real index
    
        sequence = features[start:end] # sequence
        
        # get labels     
        current_labels = kmeans.predict(sequence)
        
        # get idx --> sequence analysis
        idxMax, score = getMinIdx(current_labels,mLabels)
        scores[i] = score/sequence_length
    
        # get estimated position
        pos = trajectory_r[idxMax]
        maxPositions[i] = np.array([pos[0],pos[1],score])
    
        # distance real-estimated position
        dist = np.sqrt((pos[0]-posRef[0])**2 + (pos[1]-posRef[1])**2)
        distances[i] = dist
        
    for i in range(delete_rows):
        distances = np.delete(distances, (0), axis = 0)
        scores = np.delete(scores, (0), axis = 0)
        indices = np.delete(indices, (0), axis = 0)
        refPositions = np.delete(refPositions, (0), axis = 0)
        maxPositions = np.delete(maxPositions, (0), axis = 0)
        
    # reset number of check_points
    check_points = distances.shape[0]
    
    # Evaluation
    correct_positions = 0
    usable_positions = 0
    good_or_bad = np.zeros(distances.shape[0])
    RMSE = 0
    maxDist = 5
    for i in range(check_points):
        d = distances[i]

        if d < maxDist and traj_distances[int(indices[i])]:
            correct_positions += 1
            RMSE += d*d
            good_or_bad[i] = 1
        else:
            good_or_bad[i] = 2
        if traj_distances[int(indices[i])]:
            usable_positions += 1
        else:
            good_or_bad[i] = 0
    if correct_positions == 0:
        RMSE = 5
    else:
        RMSE = np.sqrt(RMSE/correct_positions)
    
    return correct_positions/usable_positions, RMSE
