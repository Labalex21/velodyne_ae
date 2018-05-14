# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import tflearn
import tensorflow as tf
import time
import file_handler as fh
import cv2
import math
import numpy as np
import datetime as dt
import sequence_analysis as seq
import json
from activations import lrelu # leaky rectified linear activation function

last_encoder_width = 500
number_of_conv = 2
fcs = np.array([last_encoder_width*2,last_encoder_width])

dir_test = "../data/20180201/imgs/result_ae_simple/"
dir_data = "../data/imgs/"
dir_records = "../data/20180201/records/"

# log file
log_filename = "../data/logs/log_ae_simple_" + dt.datetime.now().strftime("%Y%m%d_%H_%M_%S") + ".txt"
log_file = open(log_filename,"w")
log_file.write("start\n")
log_file.flush()

res_filename = "../data/results/ae_simple_" + dt.datetime.now().strftime("%Y%m%d_%H_%M_%S") + ".txt"
res_file = open(res_filename,"w")

# input data parameters
epochs = 20
batch_size = 20

# images parameters
max_dist = 40
height = 900
width = 16
image_shape = [height,width]
label_shape = image_shape

# network parameters
learning_rate = 0.002

n_features = 16
patch_size = 3
strides = [1, 1, 1, 1]

def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)

# define weights, uniform dis
def weight_variable(n_input, n_output, patch, name):
    W = tf.Variable(
        tf.random_uniform([patch, 1, n_input, n_output],
                          -1.0 / math.sqrt(n_input),
                          1.0 / math.sqrt(n_input)),
        name=name)
    return W

# convolutional layer
def conv(x_input, W, b,strides):
    #data input NHWC

    conv = tf.nn.conv2d(x_input, W, strides, padding='SAME') # "2d" convolution (here reduced to 1d)
    conv = tf.nn.bias_add(conv, b) # bias
    # conv = tflearn.batch_normalization(conv) # batch normalization
    conv = lrelu(conv) # leaky rectified linear activation function
    return conv

# transposed convolution for the decoder
def conv_transposed(x_input, W, output_shape, name,strides):

    b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
    output = tf.nn.conv2d_transpose(x_input, W,
                                    tf.stack([tf.shape(x)[0], output_shape[1], output_shape[2], output_shape[3]]),
                                    # output_shape[1:4],
                                    strides, padding='SAME', name=name)
    output = tf.nn.bias_add(output, b)
    # output = tflearn.batch_normalization(output)
    output = lrelu(output)
    return output

def fully_connected(x_input, W, b):
        output = tf.matmul(x_input, W)
        output = tf.add(output, b)
        # output = tflearn.batch_normalization(output)
        output = tf.nn.relu(output)
        return output

def create_network(x_input, number_fc, fc_widths):
    current_string = "create network " + str(fc_widths[0]) + " " + str(fc_widths[1]) + " " + str(fc_widths[2]) + " \n"
    log_file.write(current_string)
    log_file.flush()
    
    n_hidden_1 = int(fc_widths[0])
    n_hidden_2 = int(fc_widths[1])
    n_hidden_3 = int(fc_widths[2])
    weights = {'wconv1': weight_variable(1, n_features, patch_size, name='w_conv1_w1'),
               'wconv2': weight_variable(n_features, n_features, patch_size, name='w_conv1_w2'),
               'wconv3': weight_variable(n_features, n_features, patch_size, name='w_conv1_w3'),
               'wconv4': weight_variable(n_features, n_features, patch_size, name='w_conv1_w4'),
               'wfc1': tf.Variable(xavier_init(16 * 900 * n_features, n_hidden_1), name='w_fc1'),
               'wfc2': tf.Variable(xavier_init(n_hidden_1, n_hidden_2), name='w_fc2'),
               'wfc3': tf.Variable(xavier_init(n_hidden_2, n_hidden_3), name='w_fc3')
               }

    biases = {'b1_enc': tf.Variable(tf.zeros([n_hidden_1], dtype=tf.float32), name='encoder_b1'),
              'b2_enc': tf.Variable(tf.zeros([n_hidden_2], dtype=tf.float32), name='encoder_b2'),
              'b3_enc': tf.Variable(tf.zeros([n_hidden_3], dtype=tf.float32), name='encoder_b3'),
              'bconv1_enc': tf.Variable(tf.zeros([n_features], dtype=tf.float32), name='encoder_conv1_b1'),
              'bconv2_enc': tf.Variable(tf.zeros([n_features], dtype=tf.float32), name='encoder_conv1_b2'),
              'bconv3_enc': tf.Variable(tf.zeros([n_features], dtype=tf.float32), name='encoder_conv1_b3'),
              'bconv4_enc': tf.Variable(tf.zeros([n_features], dtype=tf.float32), name='encoder_conv1_b4'),
              'b1_dec': tf.Variable(tf.zeros([n_hidden_1], dtype=tf.float32), name='decoder_b1'),
              'b2_dec': tf.Variable(tf.zeros([n_hidden_2], dtype=tf.float32), name='decoder_b2') ,
              'b3_dec': tf.Variable(tf.zeros([16 * 900 * n_features], dtype=tf.float32), name='decoder_b3')}
              # 'b3_dec': tf.Variable(tf.zeros([2 * 113 * n_features], dtype=tf.float32), name='decoder_b3')}
    
    x = tf.reshape(x_input, [tf.shape(x_input)[0], 16, 900, 1], name='reshape_image1')
    x = tf.to_float(x) #hard code
    print('input: ', x.get_shape())

    # 1st convolution
    conv1 = conv(x, weights['wconv1'], biases['bconv1_enc'],strides)
    print('conv1: ', conv1.get_shape(),weights['wconv1'].get_shape())

    # 2nd convolution
    conv2 = conv(conv1, weights['wconv2'], biases['bconv2_enc'],strides)
    print('conv2: ', conv2.get_shape(),weights['wconv2'].get_shape())

    # # 3rd convolution
    # conv3 = conv(conv2, weights['wconv3'], biases['bconv3_enc'],strides)
    # print('conv3: ', conv3.get_shape(),weights['wconv3'].get_shape())

    # 4th convolution
    # conv4 = conv(conv3, weights['wconv4'], biases['bconv4_enc'],[1,1,1,1])
    # print('conv4: ', conv4.get_shape(),weights['wconv4'].get_shape())

    # 1st fully connected layer
    fc1 = tf.reshape(conv2, [-1, 16 * 900 * n_features])
    # fc1 = tf.reshape(conv3, [-1, 2 * 113 * n_features])
    fc1 = fully_connected(fc1, weights['wfc1'], biases['b1_enc'])
    print('fc1: ', fc1.get_shape())
    encoder = fc1

    # 2nd fully connected layer
    # fc2 = fully_connected(fc1, weights['wfc2'], biases['b2_enc'])
    # print('fc2: ', fc2.get_shape())

    # 3rd fully connected layer --> encoder values
    # fc3 = fully_connected(fc2, weights['wfc3'], biases['b3_enc'])
    # print('encoder: ', fc3.get_shape())

    # decoder starts here
    # 1st fully connected layer of decoeder
    # tfc1 = fully_connected(fc3, tf.transpose(weights['wfc3']), biases['b2_dec'])
    # print('tfc1: ', tfc1.get_shape())

    # 2nd fully connected layer of decoder
    # tfc2 = fully_connected(fc2, tf.transpose(weights['wfc2']), biases['b1_dec'])
    # print('tfc2: ', tfc2.get_shape())

    # 3rd and last fully connected layer of decoder
    tfc3 = fully_connected(fc1, tf.transpose(weights['wfc1']), biases['b3_dec'])
    tfc3 = tf.reshape(tfc3, [-1,16, 900, n_features])
    # tfc3 = tf.reshape(tfc3, [-1,2 , 113, n_features])
    print('tfc3: ', tfc3.get_shape())

    # 1st transposed convolution
    # tconv1 = conv_transposed(tfc3, W=weights['wconv4'], output_shape=conv3.get_shape().as_list(), name='tconv1',strides=[1,1,1,1])
    # print('tconv1: ', tconv1.get_shape())
    #
    # 2nd transposed convolution
    tconv2 = conv_transposed(tfc3, W=weights['wconv2'], output_shape=conv1.get_shape().as_list(), name='tconv2',strides=strides)
    print('output: ', tconv2.get_shape())

    # 3rd transposed convolution
    tconv3 = conv_transposed(tconv2, W=weights['wconv1'], output_shape=x.get_shape().as_list(), name='tconv3',strides=strides)
    output = tconv3
    print('output: ', tconv3.get_shape())

    return output, x, encoder

def train():
    print("start training...")
    current_string = "start training" + "\n"
    log_file.write(current_string)
    log_file.flush()
    with tf.Session()  as sess:
        current_string = "init global" + "\n"
        log_file.write(current_string)
        log_file.flush()
        sess.run(tf.global_variables_initializer())
        current_string = "init local" + "\n"
        log_file.write(current_string)
        log_file.flush()
        sess.run(tf.local_variables_initializer())
        
        current_string = "saver" + "\n"
        log_file.write(current_string)
        log_file.flush()
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        current_string = "Coord and threads" + "\n"
        log_file.write(current_string)
        log_file.flush()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        total_batch = int(number_batches/batch_size)
        print("total_batch:",total_batch)
        start = time.time()
        for e in range(epochs):
            print("epoch",e)
            current_string = "epoch" + str(e) + "\n"
            log_file.write(current_string)
            log_file.flush()
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
                        img_cv = np.reshape(imgs[i],[900,16,1])*255/40
                        pred_cv = np.reshape(preds[i],[900,16,1])*255/40
                        filename_input = dir_test +  str(i) + "_input.png"
                        filename_output = dir_test +  str(i)  + "_output.png"
                        cv2.imwrite(filename_input, img_cv)
                        cv2.imwrite(filename_output, pred_cv)
                            
         
        coord.request_stop()
        coord.join(threads)
        
        # Save model
        save_path = saver.save(sess, path_model)
        print("Model saved in file: %s" % save_path)

def export_encoder(path_data, path_export, path_current_traj, last_encoder_width):
    # get trajectory
    traj = fh.get_scan_trajectory(path_current_traj)
    
    # get all images
    filenames = fh.files_in_folder(path_data)
    current_string = str(filenames.shape[0]) + " files\n"
    log_file.write(current_string)
    number_of_scans = filenames.shape[0]
    
    # save feature values here
    encoder_values = np.zeros((int(number_of_scans), int(last_encoder_width)))
    k = 1
    scans_per_run = 20
    if number_of_scans % scans_per_run == 0:
        k = 0
    
    # get feature values
    saver = tf.train.Saver()
    with tf.Session()  as sess:
        #load model
        saver.restore(sess, path_model)
        for i in range(int(number_of_scans / scans_per_run) + k):
            start_idx = i * scans_per_run
            end_idx = start_idx + scans_per_run
            if end_idx > number_of_scans:
                end_idx = number_of_scans
    
            imgs = []
            for j in range(start_idx,end_idx):
                img,_ = fh.get_velodyne_img(filenames[j])
                img = img[:,:,0].transpose()
                img = np.reshape(img,[img.shape[0],img.shape[1],1])
                imgs.append(img)
            imgs = np.array(imgs)
            values = sess.run([fc], feed_dict={x: imgs})
            current_string = str(start_idx) + "-" + str(j) + " " + str(filenames[start_idx]) + "\n"
            log_file.write(current_string)
            log_file.flush()
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
                
def export_encoder_csv(path_data, path_export, path_current_traj, last_encoder_width):
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
    log_file.write(str(number_of_scans)+"\n"+str(last_encoder_width))

    # save feature values here
    encoder_values = np.zeros((int(number_of_scans), int(last_encoder_width)))
    k = 1
    scans_per_run = 20
    if number_of_scans % scans_per_run == 0:
        k = 0
        
    # get feature values
    saver = tf.train.Saver()
    with tf.Session()  as sess:
        #load model
        current_string = "Load model" + path_model + " \n"
        log_file.write(current_string)
        log_file.flush()
        saver.restore(sess, path_model)
        
        for i in range(int(number_of_scans / scans_per_run) + k):
            start_idx = i * scans_per_run
            end_idx = start_idx + scans_per_run
            if end_idx > number_of_scans:
                end_idx = number_of_scans
    
            imgs = []
            for j in range(start_idx,end_idx):
                idx = int(traj[j,0])
                img,_ = fh.get_velodyne_img_csv(filenames[idx])
                img = img.transpose()
                img = np.reshape(img,[img.shape[0],img.shape[1],1])
                imgs.append(img)
            imgs = np.array(imgs)
            values = sess.run([fc], feed_dict={x: imgs})
            current_string = str(start_idx) + "-" + str(j) + " " + str(filenames[start_idx]) + "\n"
            log_file.write(current_string)
            log_file.flush()
            values = np.array(values)
            encoder_values[start_idx:end_idx, :] = values
            
    # export values to json file
    traj = traj[:,1:3]
    with open(path_export, 'w') as f:
        json.dump({"encoder": encoder_values.tolist(), "trajectory": traj.tolist()}, f)

fc_array = np.array([1,1,1,1,1,1])
fc_size_array = np.array([[800,100,50],
                 [400,100,50],
                 [200,100,50],
                 [100,100,50],
                 [50,100,50],
                 [20,100,50]])
  
current_string = "before loop\n"
log_file.write(current_string)
log_file.flush()

for i in range(1,fc_array.shape[0]):
    
    current_string = "in loop\n"
    log_file.write(current_string)
    log_file.flush()
    number_of_fc = fc_array[i]
    path_model = "../data/20180201/models/conv_ae_velodyne_simple_" + str(fc_size_array[i,0]) + "_" + str(fc_size_array[i,1]) + "_" + str(fc_size_array[i,2]) + "_" + str(number_of_fc) + "_" + str(number_of_conv) + ".ckpt"
    #dir_test = "../data/imgs/result_ae/fc_simple/" + str(i) + "/"
    last_encoder_width = fc_size_array[i,number_of_fc-1]
    
    current_string = "reset graph\n"
    log_file.write(current_string)
    log_file.flush()
    # Reset graph
    tf.reset_default_graph()
    current_string = "read record\n"
    log_file.write(current_string)
    log_file.flush()    
    x, number_batches = fh.read_tfrecord(dir_records, image_shape, batch_size = batch_size,num_epochs=2000)
    print("number_batches: ",number_batches)
    current_string = "Number batches: " + str(number_batches) + "\n"
    log_file.write(current_string)
    log_file.flush()
    current_fc_size_array = fc_size_array[i]
    current_string = "Create network" + "\n"
    log_file.write(current_string)
    log_file.flush()
    output, x, fc = create_network(x,number_of_fc,current_fc_size_array)
    
    # loss
    loss = tf.reduce_mean(tf.pow(x - output, 2))

    # optimizer
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    current_string = "train" + "\n"
    log_file.write(current_string)
    log_file.flush()
    #train
    train()
    current_string = "Export" + "\n"
    log_file.write(current_string)
    log_file.flush()
    # export encoder    
    path_traj = '../data/traj/scan_traj_20180201_all_new.txt'
    dir_export_20180201 = '../data/features/velodyne_20180201_simple_' + str(last_encoder_width) + '_' +  str(number_of_fc) + '_' +  str(number_of_conv) + '.json'
    dir_data = '../data/20180201/scans_utm_2/'
    export_encoder(dir_data, dir_export_20180201, path_traj, last_encoder_width)

    path_traj = '../data/traj/scan_traj_20180410_2.txt'
    dir_export_20180410_2 = '../data/features/velodyne_20180410_2_simple_' + str(last_encoder_width) + '_' +  str(number_of_fc) + '_' +  str(number_of_conv) + '.json'
    dir_data = '../data/20180410/scans_rot_2/'
    export_encoder(dir_data, dir_export_20180410_2, path_traj, last_encoder_width)

    dir_export_icsens = '../data/features/velodyne_icsens_simple_' + str(last_encoder_width) + '_' +  str(number_of_fc) + '_' +  str(number_of_conv) + '.json'
    dir_data_icsens = "../data/20180201/scans_icsens/"
    path_traj_icsens = '../data/traj/scan_traj_20180201_icsens.txt'
    #export_encoder_csv(dir_data_icsens, dir_export_icsens, path_traj_icsens, last_encoder_width)
    
    dir_export_herrenhausen = '../data/features/velodyne_herrenhausen_simple_' + str(last_encoder_width) + '_' +  str(number_of_fc) + '_' +  str(number_of_conv) + '.json'
    dir_data_herrenhausen = "../data/20180206/scans/"
    path_traj_herrenhausen = '../data/traj/scan_traj_20180206.txt'
    #export_encoder_csv(dir_data_herrenhausen, dir_export_herrenhausen, path_traj_herrenhausen, last_encoder_width)

   
    #path_array_ref = [dir_export_20180201, dir_data_icsens, dir_data_herrenhausen]
    path_array_ref = [dir_export_20180201]

    # get results
    cluster_size = 500
    sequence_length = 100
    log_file.write("sequence analysis...\n")
    log_file.flush()
    compl, acc = seq.get_results(dir_export_20180410_2, path_array_ref,cluster_size,sequence_length)
    log_file.write("Done.\n")
    log_file.flush()
    current_string = "simple features: " + str(n_features) + " patch size" + str(patch_size) + " " + str(fc_size_array[i,0]) + " " + str(fc_size_array[i,1]) + " " + str(fc_size_array[i,2]) + " " + str(fc_array[i]) + " " + str(number_of_conv) + " completeness: " + str(compl) + " | RMSE: " + str(acc) + "\n"
    log_file.write(current_string)
    log_file.flush()
    res_file.write(current_string)
    res_file.flush()
