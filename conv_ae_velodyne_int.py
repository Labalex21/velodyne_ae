# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import tflearn
import tensorflow as tf
import time
import file_handler as fh
import cv2
import numpy as np
import datetime as dt
import sequence_analysis as seq
import json

last_encoder_width = 500
number_of_conv = 3
fcs = np.array([last_encoder_width*2,last_encoder_width])

dir_test = "../data/20180201/imgs/result_ae/"
dir_data = "../data/imgs/"
dir_records = "../data/20180201/records_int/"

# log file
log_filename = "../data/logs/log_ae_int_" + dt.datetime.now().strftime("%Y%m%d_%H_%M_%S") + ".txt"
log_file = open(log_filename,"w")
log_file.write("start\n")
log_file.flush()

res_filename = "../data/results/ae_int_" + dt.datetime.now().strftime("%Y%m%d_%H_%M_%S") + ".txt"
res_file = open(res_filename,"w")

# input data parameters
epochs = 20
batch_size = 100

# images parameters
max_dist = 40
height = 900
width = 16
image_shape = [height,width]
label_shape = image_shape

# network parameters
learning_rate = 0.002

n_features = 32
patch_size = 3
strides = [1, 1, 1, 1]

def create_network(x, number_fc, fc_widths):
    print('input: ',x.get_shape())
    x = tf.reshape(x[:,:,:,0], [tf.shape(x)[0], height, width, 1], name='reshape_image1')
    x = tf.to_float(x)/max_dist
    print('x:     ',x.get_shape())
    
    conv1 = tflearn.conv_2d(x,n_features,patch_size,strides, padding = 'same', activation = 'leaky_relu', name='conv1')
    print('conv1: ', conv1.get_shape())
    maxPool1 = tflearn.layers.conv.max_pool_2d (conv1, 2, padding='same')
    print('mPool1:', maxPool1.get_shape())
    
    conv2 = tflearn.conv_2d(maxPool1,n_features*2,patch_size,strides, padding = 'same', activation = 'leaky_relu', name='conv2')
    print('conv2: ', conv2.get_shape())
    maxPool2 = tflearn.layers.conv.max_pool_2d (conv2, 2, padding='same')
    print('mPool2:', maxPool2.get_shape())
    
    conv3 = tflearn.conv_2d(maxPool2,n_features*4,patch_size,strides, padding = 'same', activation = 'leaky_relu', name='conv3')
    print('conv3: ', conv3.get_shape())
    
    
    #fc1 = tflearn.fully_connected(conv3, last_encoder_width*2, activation = 'leaky_relu')
    #print('fc1: ', fc1.get_shape())
    
    #fc2 = tflearn.fully_connected(fc1, last_encoder_width, activation = 'leaky_relu')
    #print('fc1: ', fc2.get_shape())
    
    #tfc1 = tflearn.fully_connected(fc2, last_encoder_width*2, activation = 'leaky_relu')
    #print('tfc1: ', tfc1.get_shape())
    
    #tfc2 = tflearn.fully_connected(tfc1, 225*4*n_features*2, activation = 'leaky_relu')
    #tfc2 = tf.reshape(tfc2, [-1, 225, 4, n_features*2])
    #print('tfc2: ', tfc2.get_shape())
    
    fc = conv3
    for i in range(number_fc):
        fc = tflearn.fully_connected(fc, fc_widths[i], activation = 'leaky_relu')
        print('fc: ', fc.get_shape())
    
    encoder = fc
    # start decoder
    tfc = fc
    for i in range(number_fc-1):
        tfc = tflearn.fully_connected(tfc, fc_widths[number_fc-2-i], activation = 'leaky_relu')
        print('tfc: ', tfc.get_shape())
        
    tfc = tflearn.fully_connected(tfc, 225*4*n_features*2, activation = 'leaky_relu')
    tfc = tf.reshape(tfc, [-1, 225, 4, n_features*2])
    print('tfc: ', tfc.get_shape())
    
    
    tconv2 = tflearn.conv_2d_transpose(tfc,n_features*2,patch_size,maxPool2.get_shape().as_list()[1:4], padding = 'same', activation = 'leaky_relu', name='deconv2')
    print('tconv2:', tconv2.get_shape())
    
    upsample3 = tflearn.upsample_2d(tconv2,2)
    print('usamp3:', upsample3.get_shape())
    tconv3 = tflearn.conv_2d_transpose(upsample3,n_features*1,patch_size,maxPool1.get_shape().as_list()[1:4], padding = 'same', activation = 'leaky_relu', name='deconv3')
    print('tconv3:', tconv3.get_shape())
    
    upsample4 = tflearn.upsample_2d(tconv3,2)
    print('usamp4:', upsample4.get_shape())
    tconv4 = tflearn.conv_2d_transpose(upsample4,2,patch_size,x.get_shape().as_list()[1:4], padding = 'same', activation = 'leaky_relu', name='deconv4')
    print('tconv4:', tconv4.get_shape())

    output = tconv4
    print('output:', output.get_shape())

    return output, x, encoder

def train():
    print("start training...")
    with tf.Session()  as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
    
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        total_batch = int(number_batches/batch_size)
        print("total_batch:",total_batch)
        start = time.time()
        for e in range(epochs):
            print("epoch",e)
            for i in range(total_batch):
                start2 = time.time()
                _,current_loss,imgs,preds = sess.run([optimizer, loss,x, output])
                            
                elapsed = time.time() - start
                elapsed2 = time.time() - start2
                if i % 20 == 0:
                    current_string = "epoch: " + str(e+1) + " iteration: " + str(i+1) + "current los: " + str(current_loss) + "\n"
                    log_file.write(current_string)
                    log_file.flush()
                    
                    print("epoch {}/{}".format(e+1,epochs),
                          "| batch: {}/{}".format(i+1,total_batch),
                          "| current los:",current_loss,
                          "| El. time: ", "{:.2f}".format(elapsed), "s",
                          "| Batch time: ", "{:.2f}".format(elapsed2), "s")
                    
            for i in range(imgs.shape[0]):
                filename_input = dir_test +  str(i) + "_input.png"
                filename_output = dir_test +  str(i)  + "_output.png"
                cv2.imwrite(filename_input, imgs[i]*255)
                cv2.imwrite(filename_output, preds[i]*255)
            
                    
         
        coord.request_stop()
        coord.join(threads)
        
        # Save model
        save_path = saver.save(sess, path_model)
        print("Model saved in file: %s" % save_path)

def export_encoder(path_data, path_export, path_current_traj):
    # get trajectory
    traj = fh.get_scan_trajectory(path_current_traj)
    
    # get all images
    filenames = fh.files_in_folder(path_data)
    current_string = str(filenames.shape[0]) + " files\n"
    log_file.write(current_string)
    number_of_scans = filenames.shape[0]
    
    # save feature values here
    encoder_values = np.zeros((number_of_scans, last_encoder_width))
    k = 1
    if number_of_scans % 100 == 0:
        k = 0
    
    # get feature values
    saver = tf.train.Saver()
    with tf.Session()  as sess:
        #load model
        saver.restore(sess, path_model)
        
        for i in range(int(number_of_scans / 100) + k):
            start_idx = i * 100
            end_idx = start_idx + 100
            if end_idx > number_of_scans:
                end_idx = number_of_scans
    
            imgs = []
            for j in range(start_idx,end_idx):
                _,img = fh.get_velodyne_img(filenames[j])
                img = img[:,:,0]
                img = np.reshape(img,[img.shape[0],img.shape[1],1])
                imgs.append(img)
            imgs = np.array(imgs)
            current_string = str(j) + " " + str(filenames[j]) + "\n"
            log_file.write(current_string)
            log_file.flush()
            values, pred = sess.run([fc, output], feed_dict={x: imgs})
            values = np.array(values)
            encoder_values[start_idx:end_idx, :] = values
            
    # export values to json file
    with open(path_export, 'w') as f:
        json.dump({"encoder": encoder_values.tolist(), "trajectory": traj.tolist()}, f)
            
            #pred = np.array(pred)
            #pred = np.reshape(pred, [pred.shape[0], pred.shape[1], pred.shape[2]])

            #for j in range(imgs.shape[0]):
                #string_img = dir_imgs + "img_" + str(j+start_idx) + ".png"
                #string_pred = dir_pred + "img_" + str(j+start_idx) + ".png"
                #cv2.imwrite(string_img, imgs[j]*255)
                #cv2.imwrite(string_pred, pred[j]*255)
                
def export_encoder_csv(path_data, path_export, path_current_traj):
    # get trajectory
    traj = fh.get_scan_trajectory_csv(path_current_traj)
    
    # get all images
    #filenames = fh.files_in_folder(dir_data)
    filenames = fh.files_in_folder_csv(path_data)
    current_string = str(filenames.shape[0]) + " files\n"
    log_file.write(current_string)
    current_string = str(traj.shape[0]) + " scan positions\n"
    log_file.write(current_string)
    number_of_scans = traj.shape[0]
    
    # save feature values here
    encoder_values = np.zeros((number_of_scans, last_encoder_width))
    k = 1
    if number_of_scans % 100 == 0:
        k = 0
        
    # get feature values
    saver = tf.train.Saver()
    with tf.Session()  as sess:
        #load model
        saver.restore(sess, path_model)
        
        for i in range(int(number_of_scans / 100) + k):
            start_idx = i * 100
            end_idx = start_idx + 100
            if end_idx > number_of_scans:
                end_idx = number_of_scans
    
            imgs = []
            for j in range(start_idx,end_idx):
                idx = int(traj[j,0])
                _,img = fh.get_velodyne_img_csv(filenames[idx])
                img = img[:,:,0]
                img = np.reshape(img,[img.shape[0],img.shape[1],1])
                imgs.append(img)
            imgs = np.array(imgs)
            current_string = str(start_idx) + "-" + str(j) + " " + str(filenames[start_idx]) + "\n"
            log_file.write(current_string)
            log_file.flush()
            values, pred = sess.run([fc, output], feed_dict={x: imgs})
            values = np.array(values)
            encoder_values[start_idx:end_idx, :] = values
            
    # export values to json file
    traj = traj[:,1:3]
    with open(path_export, 'w') as f:
        json.dump({"encoder": encoder_values.tolist(), "trajectory": traj.tolist()}, f)

fc_array = np.array([1,1,2,2,3,3])
fc_size_array = np.array([[last_encoder_width,0,0],
                 [last_encoder_width*2,0,0],
                 [last_encoder_width*2,last_encoder_width,0],
                 [last_encoder_width,last_encoder_width/2,0],
                 [last_encoder_width*2,last_encoder_width,last_encoder_width/2],
                 [last_encoder_width,last_encoder_width/2,100]])
        
for i in range(fc_array.shape[0]):
    number_of_fc = fc_array[i]
    path_model = "../data/20180201/models/conv_ae_velodyne_int_" + str(fc_size_array[i,0]) + "_" + str(fc_size_array[i,1]) + "_" + str(fc_size_array[i,2]) + "_" + str(number_of_fc) + "_" + str(number_of_conv) + ".ckpt"
    dir_test = "../data/imgs/result_ae/fc/" + str(i) + "/"
    
    # Reset graph
    tf.reset_default_graph()
        
    x, number_batches = fh.read_tfrecord(dir_records, image_shape, batch_size = batch_size,num_epochs=2000)
    print("number_batches: ",number_batches)

    current_fc_size_array = fc_size_array[i,0:fc_array[i]]
    output, x, fc = create_network(x,number_of_fc,current_fc_size_array)

    # loss
    loss = tf.reduce_mean(tf.pow(x - output, 2))

    # optimizer
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

    #train
    train()

    # export encoder    
    path_traj = '../data/traj/scan_traj_20180201.txt'
    dir_export_20180201 = '../data/features/velodyne_20180201_' + str(last_encoder_width) + '_' +  str(number_of_fc) + '_' +  str(number_of_conv) + '.json'
    dir_data = '../data/20180201/scans_csv/'
    export_encoder_csv(dir_data, dir_export_20180201, path_traj)

    path_traj = '../data/traj/scan_traj_20180410_2.txt'
    dir_export_20180410_2 = '../data/features/velodyne_20180410_2_' + str(last_encoder_width) + '_' +  str(number_of_fc) + '_' +  str(number_of_conv) + '.json'
    dir_data = '../data/20180410/scans_rot_2/'
    export_encoder(dir_data, dir_export_20180410_2, path_traj)

    dir_export_icsens = '../data/features/velodyne_icsens_' + str(last_encoder_width) + '_' +  str(number_of_fc) + '_' +  str(number_of_conv) + '.json'
    dir_data_icsens = "../data/20180201/scans_icsens/"
    path_traj_icsens = '../data/traj/scan_traj_20180201_icsens.txt'
    #export_encoder_csv(dir_data_icsens, dir_export_icsens, path_traj_icsens)
    
    dir_export_herrenhausen = '../data/features/velodyne_herrenhausen_' + str(last_encoder_width) + '_' +  str(number_of_fc) + '_' +  str(number_of_conv) + '.json'
    dir_data_herrenhausen = "../data/20180206/scans/"
    path_traj_herrenhausen = '../data/traj/scan_traj_20180206.txt'
    #export_encoder_csv(dir_data_herrenhausen, dir_export_herrenhausen, path_traj_herrenhausen)

   
    #path_array_ref = [dir_export_20180201, dir_data_icsens, dir_data_herrenhausen]
    path_array_ref = [dir_export_20180201]

    # get results
    cluster_size = 200
    sequence_length = 200
    compl, acc = seq.get_results(dir_export_20180201, path_array_ref,cluster_size,sequence_length)
    current_string = "features: " + str(n_features) + " patch size" + str(patch_size) + " " + str(fc_size_array[i,0]) + " " + str(fc_size_array[i,1]) + " " + str(fc_size_array[i,2]) + " " + str(fc_array[i]) + "_" + str(number_of_conv) + " completeness: " + str(compl) + " | RMSE: " + str(acc) + "\n"
    log_file.write(current_string)
    res_file.write(current_string)
