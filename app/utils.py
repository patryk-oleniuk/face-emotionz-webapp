import sys
# IMPORTANT: OPENCV 3 for Python 3 is needed, install it from : 
# http://docs.opencv.org/3.0-beta/doc/tutorials/introduction/windows_install/windows_install.html
# or on MAC : brew install opencv3 --with-contrib --with-python3 --HEAD
# http://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/

import cv2
import matplotlib.pyplot as plt
import PIL 
import numpy as np
import tensorflow.compat.v1 as tf

from scipy.sparse import coo_matrix

# returns the array of 48x48 images of faces and the whole image with rectangles over the faces img_path = 'camera' or 'file.png' or 'file.jpg' 
def get_faces_from_img(image):
    
    # The face recognition properties, recognizing only frontal face
    cascPath = 'opencv-artifacts/haarcascade_frontalface_default.xml'
    
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    #convert to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    #print("Found {0} faces in ".format(len(faces)), img_path, " !")

    #preparing an array to store each face img separately 
    faces_imgs = np.zeros((len(faces),48,48))

    # iterate through the faces and save them into a separate array
    num_fac = 0

    for (x, y, w, h) in faces:

        face_single = image[y:y+h,x:x+w]
        #resize to 48x48
        face_resized = cv2.resize(face_single, (48,48))
        #cv2.imwrite('Face'+str(num_fac)+'.png', face_resized)
        #taking only one color (because it's grey RGB)
        faces_imgs[num_fac] = face_resized[:,:,0]
        num_fac = num_fac+1
        #adding rectangles to faces

    # adding rectangles on faces in the image
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
    #cv2.imshow("Faces found", image)
    #cv2.imwrite('Faces_recognized.png', image)
    return faces_imgs, image

def convert_to_one_hot(a,max_val=None):
    N = a.size
    data = np.ones(N,dtype=int)
    sparse_out = coo_matrix((data,(np.arange(N),a.ravel())), shape=(N,max_val))
    return np.array(sparse_out.todense())

#convert single vector to a value
def convert_from_one_hot(a):
     return np.argmax(a);
    
def get_min_max( img ):
    return np.min(np.min(img)),np.max(np.max(img))

def remap(img, min, max):
    if ( max-min ):
        return (img-min) * 1.0/(max-min)
    else:
        return img
    
#constrast stretch function
def contrast_stretch(img ):
    min, max = get_min_max( img );
    return remap(img, min, max)

#calculating partial accuracy (for each clas separately)
def calc_partial_accuracy(tset, result, emlabel):
    
    tsetlabels = np.where(tset == emlabel)[0];
    resultlabels = np.where(result == emlabel)[0];

    errors =0;
    for label in resultlabels :
        if label not in tsetlabels:
            errors += 1;
    
    for label in tsetlabels :
        if label not in resultlabels:
            errors += 1;
    
    return (len(resultlabels)+ len(tsetlabels)- errors)/ (len(resultlabels)+ len(tsetlabels))

# loads the  csv labelled emotion images dataset 
def load_dataset(reader, num_data, hist_div, hist_threshold):
    #preparing arrays
    emotions = np.zeros(num_data)
    images = np.zeros((num_data,48,48))
    strange_im = np.zeros((int(num_data/10),48,48)) # the dataset contains <10% of strange img

    # for image pre-filtering
    num_strange = 0; #number of removed images
    num_skipped = 0; #hapy images skip counter
    rownum =0;
    #parsing each row
    for row in reader:
        #(column0) extract the emotion label
        #!!!! convert 1 and 0 together !!!!
        if( (row[0] == '0') or (row[0] == '1' ) ):
            emotions[rownum] = '0';
        else :
            emotions[rownum] = str(int(row[0])-1)

        #ignore 1/3 of happy cklass pic, there are too many in relative to to others  
        if( (emotions[rownum] != 2 ) or ((emotions[rownum] == 2) and (np.random.choice([0,1,1]) == 1) )): 

            #(column1) extract the image data, parse it and convert into 48x48 array of integers
            images[rownum] = np.asarray([int(s) for s in row[1].split(' ')]).reshape(48,48)

            #stretching contrast of the image
            images[rownum] = contrast_stretch(images[rownum])

            #calculating the histogram and erasing "strange" images
            y_h, x_h = np.histogram( images[ rownum ] , 100 );
            if y_h.max() > hist_threshold  : 
                # if img is 'strange'
                strange_im[num_strange,:,:] = images[rownum,:,:];
                num_data = num_data - 1;
                images = np.delete(images, rownum, axis = 0);
                emotions = np.delete(emotions, rownum)
                #print('deleted:',rownum, y_h.max())
                num_strange += 1;   
            else:
                rownum += 1
            if not rownum%500:
                print("loaded %2.0f" % ((float(rownum ) /num_data)*100) 
                      + '% of dataset ('+ str(rownum+num_strange)+'/'+ str(num_data) + '). Filtered images: ' + str(num_strange) )
        else:
            images = np.delete(images, rownum, axis = 0);
            emotions = np.delete(emotions, rownum)
            num_skipped +=1; # skip some happy images 
    
    return images, emotions, strange_im, num_strange, num_skipped


def set_tf_model_graph(nr_max_faces):
    d = 2304 #train_data.shape[1]

    def weight_variable2(shape, nc10):
            initial2 = tf.random_normal(shape, stddev=tf.sqrt(2./tf.to_float(ncl0)) )
            return tf.Variable(initial2)
        
    def conv2dstride2(x,W):
            return tf.nn.conv2d(x,W,strides=[1, 2, 2, 1], padding='SAME')

    def conv2d(x,W):
            return tf.nn.conv2d(x,W,strides=[1, 1, 1, 1], padding='SAME')
        
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=1/np.sqrt(d/2) )
            return tf.Variable(initial)
        
    def bias_variable(shape):
        initial = tf.constant(0.01,shape=shape)
        return tf.Variable(initial)
    
    tf.reset_default_graph()

    # implementation of Conv-Relu-COVN-RELU - pool
    # based on : http://cs231n.github.io/convolutional-networks/

    # Define computational graph (CG)
    batch_size = nr_max_faces    # batch size
    nc = 6                  # number of classes

    # Inputs
    xin = tf.placeholder(tf.float32,[batch_size,d]); #print('xin=',xin,xin.get_shape())
    y_label = tf.placeholder(tf.float32,[batch_size,nc]); #print('y_label=',y_label,y_label.get_shape())


    #for the first conc-conv
    # Convolutional layer
    K0 = 8   # size of the patch
    F0 = 22  # number of filters
    ncl0 = K0*K0*F0

    #for the second conc-conv
    K1 = 4   # size of the patch
    F1 = F0  # number of filters
    ncl1 = K1*K1*F1

    #drouput probability
    keep_prob_input=tf.placeholder(tf.float32)

    #1st set of conv followed by conv2d operation and dropout 0.5
    W_conv1=weight_variable([K0,K0,1,F0])
    b_conv1=bias_variable([F0])
    x_2d1 = tf.reshape(xin, [-1,48,48,1])

    #conv2d 
    h_conv1=tf.nn.relu(conv2d(x_2d1, W_conv1) + b_conv1)
    #h_conv1= tf.nn.dropout(h_conv1,keep_prob_input);

    # 2nd convolutional layer + max pooling
    W_conv2=weight_variable([K0,K0,F0,F0])
    b_conv2=bias_variable([F0])

    # conv2d + max pool
    h_conv2 = tf.nn.relu(conv2d(h_conv1,W_conv2)+b_conv2)
    h_conv2_pooled = max_pool_2x2(h_conv2)

    #3rd set of conv 
    W_conv3=weight_variable([K0,K0,F0,F0])
    b_conv3=bias_variable([F1])
    x_2d3 = tf.reshape(h_conv2_pooled, [-1,24,24,F0])

    #conv2d
    h_conv3=tf.nn.relu(conv2d(x_2d3, W_conv3) + b_conv3)

    # 4th convolutional layer 
    W_conv4=weight_variable([K1,K1,F1,F1])
    b_conv4=bias_variable([F1])

    #conv2d + max pool 4x4
    h_conv4 = tf.nn.relu(conv2d(h_conv3,W_conv4)+b_conv4)
    h_conv4_pooled = max_pool_2x2(h_conv4)
    h_conv4_pooled = max_pool_2x2(h_conv4_pooled)

    #5th set of conv 
    W_conv5=weight_variable([K1,K1,F1,F1])
    b_conv5=bias_variable([F1])
    x_2d5 = tf.reshape(h_conv4_pooled, [-1,6,6,F1])

    #conv2d
    h_conv5=tf.nn.relu(conv2d(x_2d5, W_conv5) + b_conv5)

    # 6th convolutional layer 
    W_conv6=weight_variable([K1,K1,F1,F1])
    b_conv6=bias_variable([F1])
    b_conv6= tf.nn.dropout(b_conv6,keep_prob_input)

    #conv2d + max pool 4x4
    h_conv6 = tf.nn.relu(conv2d(h_conv5,W_conv6)+b_conv6)
    h_conv6_pooled = max_pool_2x2(h_conv6)

    # reshaping for fully connected
    h_conv6_pooled_rs = tf.reshape(h_conv6, [batch_size,-1])
    W_norm6 = weight_variable([  6*6*F1, nc])
    b_norm6 = bias_variable([nc])

    # fully connected layer
    h_full6 = tf.matmul( h_conv6_pooled_rs, W_norm6 )
    h_full6 += b_norm6

    y = h_full6; 

    ## Softmax
    y = tf.nn.softmax(y)

    # Loss
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(y), 1))
    total_loss = cross_entropy

    # Optimization scheme
    train_step = tf.train.AdamOptimizer(0.001).minimize(total_loss)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_label,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return y, xin, keep_prob_input