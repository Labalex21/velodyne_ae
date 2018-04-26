# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 16:38:14 2017

@author: schlichting
"""

import numpy as np
import matplotlib.pyplot as plt
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
    
def checkMax(idx):
    if idx <= trajectory.shape[0]-trajectory3.shape[0] or idx <= sequence_length:
        return True
    # check for distance jump
    for i in range(sequence_length):
        if getDist2(trajectory[idx-i],trajectory[idx-i-1]) > 20**2:
            return False
    return True
    
def filter_by_tree(tree, traj, features):
    distances = tree.query_radius(traj[:,0:2],r = 10,count_only=True)
    traj = traj[(distances == 0)]
    features = features[(distances == 0)]
    return features, traj

def sample_data(data,traj,seg_dist=0.3):
    # f = 12.5 Hz --> every 0.08 s
    n = traj.shape[0]
    data_new = []
    traj_new =[]
    dist_all = 0
    #dist_sum = 0
    for i in range(n-1):
        dist = np.linalg.norm(traj[i]-traj[i+1])
        dist_all += dist
        #dist_sum += dist
        if dist_all > seg_dist:
            dist_all -= seg_dist
            traj_new.append(traj[i])
            data_new.append(data[i])
    data_new = np.array(data_new)
    traj_new = np.array(traj_new)
    #print("traj dist: ", dist_sum,' m.')
    return data_new, traj_new

def analyze_results():
    diffs = np.linalg.norm(maxPositions[:,0:2]-refPositions, axis = 1);
    diffs[diffs > 5] = 5
    plt.hist(maxPositions[:,2])
    #plt.plot(maxPositions[:,2],diffs,'.')


def get_results(path_online, path_ref_vector, cluster_size, sequence_length):
    print("load data")
    
    reference, trajectory_r = import_encoder_traj(path_ref_vector[0])
    print("create tree")
    tree_ref = KDTree(trajectory_r[:,0:2])
    for i in range(1,len(path_ref_vector)):
        reference_tmp, trajectory_r_tmp = import_encoder_traj(path_ref_vector[i])
        print(reference_tmp.shape)
        reference_tmp, trajectory_r_tmp = filter_by_tree(tree_ref, trajectory_r_tmp,reference_tmp)
        print(reference_tmp.shape)
        reference = np.concatenate((reference,reference_tmp), axis=0)
        trajectory_r = np.concatenate((trajectory_r,trajectory_r_tmp), axis=0)
    features, trajectory_f = import_encoder_traj(path_online)
    
    print("sample data")
    reference, trajectory_r = sample_data(reference,trajectory_r)
    features, trajectory_f = sample_data(features,trajectory_f)
    
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



#correct_positions, rmse = get_results('data/encoder_1457_224.csv',['data/encoder_1044_224.csv','data/encoder_1457_224.csv'],200,200)

#plt.close("all")
#
##number_features_vector = [10,20,50,100,150,'ranges']
##number_features_vector = [50,100,150,'ranges']
#number_features_vector = [224]
#results_features = []
#results_features_rms = []
#
#for number_features in number_features_vector:
#    
#    
#    #number_features = 'ranges'
#    #number_features = 224
##    number_features = 224
#    path1044 = 'data/encoder_1044_' + str(number_features) + ".csv"
#    path1457 = 'data/encoder_1457_' + str(number_features) + ".csv"
#    path1457_2 = 'data/encoder_1457_2_' + str(number_features) + ".csv"
#    pathIcsens = 'data/encoder_icsens_' + str(number_features) + ".csv"
##    pathRobotcar232 = 'data/robotcar/encoder_20140623153604_' + str(number_features) + '.csv'
##    pathRobotcar24 = 'data/robotcar/encoder_20140624_' + str(number_features) + '.csv'
##    pathRobotcar232_int = 'data/robotcar/encoder_int_20140623153604_' + str(number_features) + '.csv'
##    pathRobotcar24_int = 'data/robotcar/encoder_int_20140624_' + str(number_features) + '.csv'
#    
#
#    
#    
#    print("Load data...")
##    reference, trajectory = import_encoder_traj(pathRobotcar24)
##    features, trajectory_f = import_encoder_traj(pathRobotcar232)
##    reference_int, trajectory_int = import_encoder_traj(pathRobotcar24_int)
##    features_int, trajectory_f_int = import_encoder_traj(pathRobotcar232_int)
#    
##    features, trajectory_f = import_encoder_traj(pathRobotcar24)
#    reference1, trajectory1 = import_encoder_traj(path1044)
#    reference2, trajectory2 = import_encoder_traj(pathIcsens)
#    reference3, trajectory3 = import_encoder_traj(path1457_2)
#    features, trajectory_f = import_encoder_traj(path1457)
#    #
##    reference, trajectory = sample_data(reference,trajectory)
##    features, trajectory_f = sample_data(features,trajectory_f)
##    reference_int, trajectory_int = sample_data(reference_int,trajectory_int)
##    features_int, trajectory_f_int = sample_data(features_int,trajectory_f_int)
##    reference = np.concatenate((reference,reference_int), axis=1)
##    features = np.concatenate((features,features_int), axis=1)
#    
##    lw = 3.0
##    plt.figure(figsize=(8.5,9))
##    plt.plot(trajectory[0:trajectory.shape[0]:10,0],trajectory[0:trajectory.shape[0]:10,1], c="blue",ls="solid",linewidth=lw, label='Reference')
##    ##    plt.plot(trajectory1[0:trajectory1.shape[0]:100,0],trajectory1[0:trajectory1.shape[0]:100,1], c="blue",ls="solid",linewidth=lw, label='Reference')
##    ##plt.plot(trajectory2[0:trajectory2.shape[0]:100,0],trajectory2[0:trajectory2.shape[0]:100,1], c="blue",ls="solid",linewidth=lw)
##    plt.plot(trajectory_f[0:trajectory_f.shape[0]:10,0],trajectory_f[0:trajectory_f.shape[0]:10,1], c="red",ls="solid",linewidth=lw, label='Trajectory')
#    
#    #
#    ##    plt.axis('equal') 
#    ##plt.axis([544000, 551000, 5804000, 5807000])
#    ##plt.xticks(np.arange(547000, 551000+1, 1000))
#    ##plt.yticks(np.arange(5799000, 5805000+1, 1000))
#    ##ax = plt.gca()
#    ##ax.ticklabel_format(useOffset=False)
##    plt.xlabel('East [m]')
##    plt.ylabel('North [m]')
##    plt.axis('equal')
#    #
#    #
#    reference = np.concatenate((reference1,reference2), axis=0)
#    trajectory = np.concatenate((trajectory1,trajectory2), axis=0)
#    print("Create kdtree...")
#    tree = KDTree(trajectory[:,0:2])
#    reference = np.concatenate((reference,reference3), axis=0)
#    trajectory = np.concatenate((trajectory,trajectory3), axis=0)
#    #
#    #
##    cluster_size_vector = [2,5,10,20,50,100,300,400,500]
##    cluster_size_vector = [10,20,50,100,300,400,500]
#    cluster_size_vector = [200]
#    for cluster_size in cluster_size_vector:
#        
#        pathKmeans = 'data/kmeans' + str(cluster_size) + '_' + str(number_features) + "_2.pckl"
##        pathKmeans = 'data/kmeans' + str(cluster_size) + '_' + str(number_features) + "_rcs2.pckl"
#        print("Cluster (kmeans) with k =", cluster_size,"...")
#        
#        ## create cluster
#        if os.path.isfile(pathKmeans):
#            f = open(pathKmeans, 'rb')
#            kmeans = pickle.load(f)
#            f.close()
#            print("Loaded cluster")
#        else:
#            kmeans = KMeans(n_clusters=cluster_size).fit(reference)
#            f = open(pathKmeans, 'wb')
#            pickle.dump(kmeans, f, 2)
#            f.close
#            print("Saved cluster.")
#    
#        # load cluster
#        f = open(pathKmeans, 'rb')
#        kmeans = pickle.load(f)
#        f.close()
#        
#        labels = kmeans.labels_
#        
#        print("Get trajectory distances...")
#        traj_distances = tree.query_radius(trajectory_f[:,0:2],r = 5.,count_only=True)
#        
#        # set sequence length
##        sequence_length = 50
#        #sequence_length_vector = [300,400,500]
#        #sequence_length_vector = [2,5,10,20,50,100,200,300,400,500]
#        sequence_length_vector = [200]
#        results_sequence = []
#        results_sequence_rms = []
#        for sequence_length in sequence_length_vector:
#                
#            # get label matrix for sequence analysis
#            mLabels = getLabelMatrix(labels,sequence_length)
#            
#            
#            #check_points = np.array(features).shape[0]-sequence_length # All points
#            check_points = 5000
#            
#            distances = np.zeros(check_points)
#            scores = np.zeros(check_points)
#            indices = np.zeros(check_points)
#            refPositions = np.zeros([check_points,2])
#            maxPositions = np.zeros([check_points,3])
#            
#            time_overall = 0
#            delete_rows = 0
#            #for i in range(features.shape[0]-number_of_features):
#            #    start = i
#            print("Sequence analysis...")
#            for i in range(0,check_points):
#                end = int(features.shape[0]/check_points)*i
#                start = end-sequence_length
#            
#                if i%500 == 0 and i  != 0:
#                    print(i,"/",check_points)
#                
#                # check if last n (sequence length) positions are available
#                if start < 0:
#                    delete_rows += 1
#                    continue
#                
#                posRef = trajectory_f[end] # Real position
#                refPositions[i] = posRef[0:2]
#                indices[i] = end # Real index
#            
#                sequence = features[start:end] # sequence
#            
#                # set start time
#                startTime = time.time()
#                
#                # get labels
#                start = time.time()        
#                current_labels = kmeans.predict(sequence)
#                elapsed = time.time() - start
##                print(elapsed, " s")   
#                
#                # get idx --> sequence analysis
#                idxMax, score = getMinIdx(current_labels,mLabels)
#                scores[i] = score/sequence_length
#                
#                elapsed = time.time() - startTime
#                time_overall += elapsed
#            
#                # get estimated position
#                pos = trajectory[idxMax]
#                maxPositions[i] = np.array([pos[0],pos[1],score])
#            
#                # distance real-estimated position
#                dist = np.sqrt((pos[0]-posRef[0])**2 + (pos[1]-posRef[1])**2)
#                distances[i] = dist
#                
#            for i in range(delete_rows):
#                distances = np.delete(distances, (0), axis = 0)
#                scores = np.delete(scores, (0), axis = 0)
#                indices = np.delete(indices, (0), axis = 0)
#                refPositions = np.delete(refPositions, (0), axis = 0)
#                maxPositions = np.delete(maxPositions, (0), axis = 0)
#                
#            # reset number of check_points
#            check_points = distances.shape[0]
#                    
#            #plt.figure()
#            #plt.plot(trajectory_f[0:trajectory_f.shape[0]:1,0],trajectory_f[0:trajectory_f.shape[0]:1,1], c="green",ls="solid", label='Trajectory')
#            #plt.plot(trajectory[0:trajectory.shape[0]:1,0],trajectory[0:trajectory.shape[0]:1,1], c="black",ls="dashed", label='Map')
#            #
#            #i = 0
#            #for i in range(maxPositions.shape[0]-1):
#            #    plt.plot(maxPositions[i,0],maxPositions[i,1],'x',c="blue", ms=10, mew = 2)
#            #    plt.plot(refPositions[i,0],refPositions[i,1],'x',c="red", ms=6, mew = 2)
#            #    plt.text(maxPositions[i,0]+2,maxPositions[i,1],str(i+1),color="blue")
#            #    plt.text(refPositions[i,0]+2,refPositions[i,1],str(i+1),color="red")
#            #
#            #i += 1    
#            #plt.plot(maxPositions[i,0],maxPositions[i,1],'x',c="blue", ms=10, mew = 2, label='Real position')
#            #plt.plot(refPositions[i,0],refPositions[i,1],'x',c="red", ms=6, mew = 2, label='Est. position')
#            #plt.text(maxPositions[i,0]+2,maxPositions[i,1],str(i+1),color="blue")
#            #plt.text(refPositions[i,0]+2,refPositions[i,1],str(i+1),color="red")
#            #ax = plt.gca()
#            #ax.set_xlim([np.min(trajectory_f[:,0])-50,np.max(trajectory_f[:,0])+50])
#            #ax.set_ylim([np.min(trajectory_f[:,1])-50,np.max(trajectory_f[:,1])+50])
#            #plt.legend(numpoints=1)
#            
#            
#            # Evaluation
#            correct_positions = 0
#            usable_positions = 0
#            good_or_bad = np.zeros(distances.shape[0])
#            RMSE = 0
#            maxDist = 10
#            for i in range(check_points):
#                d = distances[i]
#            #    print(i+1,d)
#                if d < maxDist and traj_distances[int(indices[i])]:
#                    correct_positions += 1
#                    RMSE += d*d
#                    good_or_bad[i] = 1
#                else:
#                    good_or_bad[i] = 2
#                if traj_distances[int(indices[i])]:
#                    usable_positions += 1
#                else:
#                    good_or_bad[i] = 0
#            if correct_positions == 0:
#                RMSE = 5
#            else:
#                RMSE = np.sqrt(RMSE/correct_positions)
#            
#            print(features.shape[1],"Features, cluster_size, clusters = ", cluster_size," seqence length = ", sequence_length,":", correct_positions/usable_positions*100, "%, RMSE:", RMSE, "m, Mean time:", time_overall/check_points,"s")
#            results_sequence.append(correct_positions/usable_positions)
#            results_sequence_rms.append(RMSE)
#        results_sequence = np.array(results_sequence)
#        results_features.append(results_sequence)
#        results_sequence_rms = np.array(results_sequence_rms)
#        results_features_rms.append(results_sequence_rms)
#results_features = np.array(results_features)
#results_features_rms = np.array(results_features_rms)