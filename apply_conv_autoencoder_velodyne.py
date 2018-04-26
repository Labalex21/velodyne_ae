# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import tflearn
import tensorflow as tf
import json
import file_handler as fh
import cv2
import numpy as np
import datetime as dt

# Reset graph
tf.reset_default_graph()

last_encoder_width = 500

#dir_data = "D:/Velodyne/20180201_icsens_innenstadt/imgs/"
#dir_test = "D:/Velodyne/20180201_icsens_innenstadt/imgs/result_ae/"
dir_data = "../data/scans_all/"
dir_imgs = "../data/imgs/ae_input/"
dir_pred = "../data/imgs/ae_pred/"
dir_records = "../data/imgs/records/"
#path_model = "D:/Velodyne/20180201_icsens_innenstadt/models/conv_dyn_velodyne.ckpt"
path_model = "../data/models/conv_ae_velodyne.ckpt"
path_traj = '../data/traj/scan_traj_20180201.txt'
dir_export = '../data/features/velodyne_' + str(last_encoder_width) + '.json'

# log file
log_filename = "../data/logs/log_features_" + dt.datetime.now().strftime("%Y%m%d_%H_%M_%S") + ".txt"
#log_filename = "D:/Velodyne/20180201_icsens_innenstadt/logs/log_ae_" + dt.datetime.now().strftime("%Y%m%d_%H_%M_%S") + ".txt"
log_file = open(log_filename,"w")
log_file.write("start\n")
log_file.flush()

# input data parameters
epochs = 200
batch_size = 100

# images parameters
max_dist = 40
height = 900
width = 16
image_shape = [height,width]
label_shape = image_shape

# network parameters
keep_prob = 0.5
learning_rate = 0.002

n_features = 32
patch_size = 3
strides = [1, 1, 1, 1]

def create_network(x, number_fc, fc_widths):
    print('input: ',x.get_shape())
    x = tf.reshape(x, [tf.shape(x)[0], height, width, 1], name='reshape_image1')
    print(x)
    x = tf.to_float(x)/max_dist
    print(x)
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
#    maxPool3 = tflearn.layers.conv.max_pool_2d (conv3, 2, padding='same')
#    print('mPool3:', maxPool3.get_shape())
    
#    conv4 = tflearn.conv_2d(maxPool3,n_features*4,patch_size,strides, padding = 'same', activation = 'leaky_relu')
#    print('conv4: ', conv4.get_shape())
#    maxPool4 = tflearn.layers.conv.max_pool_2d (conv4, 2, padding='same')
#    print('mpool4:', conv4.get_shape())
#    
#
    fc = conv3
    for i in range(number_fc):
        fc = tflearn.fully_connected(fc, fc_widths[i], activation = 'leaky_relu')
        print('fc: ', fc.get_shape())
       
    tfc = fc
    for i in range(number_fc-1):
        tfc = tflearn.fully_connected(tfc, fc_widths[number_fc-2-i], activation = 'leaky_relu')
        print('tfc: ', tfc.get_shape())
        
    tfc = tflearn.fully_connected(tfc, 225*4*n_features*2, activation = 'leaky_relu')
    tfc = tf.reshape(tfc, [-1, 225, 4, n_features*2])
    print('tfc: ', tfc.get_shape())

        #fc2 = tflearn.fully_connected(fc1, last_encoder_width, activation = 'leaky_relu')
        #print('fc2: ', fc2.get_shape())
    
    #tfc1 = tflearn.fully_connected(fc2, last_encoder_width*2, activation = 'leaky_relu')
    #print('tfc1: ', tfc1.get_shape())
    
    #tfc2 = tflearn.fully_connected(tfc1, 225*4*n_features*2, activation = 'leaky_relu')
    #tfc2 = tf.reshape(tfc2, [-1, 225, 4, n_features*2])
    #print('tfc2: ', tfc2.get_shape())
    
#    last = fully_connected(tfc2, tf.transpose(weights['wfc1']), biases['b3_dec'])
#    # tfc2 = tf.reshape(tfc2, [-1, 1160*2, 1, n_features])
#    last = tf.reshape(last, [-1, 1160, 1, n_features])
#    print('tfc3: ', last.get_shape())
    
#    upsample1 = tflearn.upsample_2d(maxPool4,2)
#    print('usamp1:', upsample1.get_shape())
#    tconv1 = tflearn.conv_2d_transpose(fc1,n_features*4,patch_size,maxPool3.get_shape().as_list()[1:4], padding = 'same', activation = 'leaky_relu')
#    print('tconv1:', tconv1.get_shape())
    
#    upsample2 = tflearn.upsample_2d(tconv1,2)
#    print('usamp2:', upsample2.get_shape())
    tconv2 = tflearn.conv_2d_transpose(tfc,n_features*2,patch_size,maxPool2.get_shape().as_list()[1:4], padding = 'same', activation = 'leaky_relu', name='deconv2')
    print('tconv2:', tconv2.get_shape())
    
    upsample3 = tflearn.upsample_2d(tconv2,2)
    print('usamp3:', upsample3.get_shape())
    tconv3 = tflearn.conv_2d_transpose(upsample3,n_features*1,patch_size,maxPool1.get_shape().as_list()[1:4], padding = 'same', activation = 'leaky_relu', name='deconv3')
    print('tconv3:', tconv3.get_shape())
    
    upsample4 = tflearn.upsample_2d(tconv3,2)
    print('usamp4:', upsample4.get_shape())
    tconv4 = tflearn.conv_2d_transpose(upsample4,1,patch_size,x.get_shape().as_list()[1:4], padding = 'same', activation = 'leaky_relu', name='deconv4')
    print('tconv4:', tconv4.get_shape())

    output = tconv4
    print('output:', output.get_shape())

    return output, x, fc

def export_encoder():
    # get trajectory
    traj = fh.get_scan_trajectory(path_traj)
    
    # get all images
    filenames = fh.files_in_folder(dir_data)
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
                img,_ = fh.get_velodyne_img(filenames[j])
                img = img[:,:,0]
                #img = np.reshape(img,[img.shape[0],img.shape[1],1])
                imgs.append(img)
            imgs = np.array(imgs)
            current_string = str(j) + " " + str(filenames[j]) + "\n"
            log_file.write(current_string)
            log_file.flush()
            values, pred = sess.run([encoder, output], feed_dict={x: imgs})
            values = np.array(values)
            encoder_values[start_idx:end_idx, :] = values
            
    # export values to json file
    with open(dir_export, 'w') as f:
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
    print(traj[0])
    
    # get all images
    #filenames = fh.files_in_folder(dir_data)
    filenames = fh.files_in_folder_csv(dir_data_icsens)
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
                img,_ = fh.get_velodyne_img_csv(filenames[idx])
                img = img[:,:,0]
                #img = np.reshape(img,[img.shape[0],img.shape[1],1])
                imgs.append(img)
            imgs = np.array(imgs)
            current_string = str(start_idx) + "-" + str(j) + " " + str(filenames[start_idx]) + "\n"
            log_file.write(current_string)
            log_file.flush()
            values, pred = sess.run([encoder, output], feed_dict={x: imgs})
            values = np.array(values)
            encoder_values[start_idx:end_idx, :] = values
            
    # export values to json file
    traj = traj[:,1:3]
    with open(dir_export, 'w') as f:
        json.dump({"encoder": encoder_values.tolist(), "trajectory": traj.tolist()}, f)
                

x, number_batches = fh.read_tfrecord(dir_records, image_shape, batch_size = batch_size,num_epochs=epochs)

output, x, encoder = create_network(x,2,np.array([last_encoder_width*2,last_encoder_width]))

# loss
loss = tf.reduce_mean(tf.pow(x - output, 2))

# optimizer
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

dir_export_icsens = '../data/features/velodyne_icsens_' + str(last_encoder_width) + '.json'
dir_data_icsens = "../data/20180201/scans_icsens/"
path_traj_icsens = '../data/traj/scan_traj_20180201_icsens.txt'
export_encoder_csv(dir_data_icsens, dir_export_icsens, path_traj_icsens)
#export_encoder()
log_file.close()
