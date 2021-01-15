import os

import numpy as np
import streamlit as st
import random
import importlib  
from webcam import webcam
from PIL import Image 

import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import utils as ut

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

tf.compat.v1.disable_eager_execution()

nr_max_faces=20
nc=6

str_emotions = ['angry','scared','happy','sad','surprised','normal']

y, xin, keep_prob_input = ut.set_tf_model_graph(nr_max_faces)
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, 'mlmodels/face-emotion-recognition/model_6layers.ckpt')

st.title("Facial Emotion Recognizer")
st.subheader("&#8592; Choose the image source on the sidebar:")
st.markdown("test image, webcam, or upload")
st.subheader('''First, OpenCV will detect faces.''')
st.subheader('''Then, Tensorflow Deep Neural Network will recognize their emotions.''')

source = st.sidebar.selectbox( "Image Source? ", ('Test Image', 'Upload', 'Webcam') )

if source == 'Webcam':
    captured_image = webcam()
elif source == 'Upload':
    file = st.file_uploader("Upload your img file here (.jpg or .png)")
    if file is not None:
        captured_image = Image.open(file)
    else :
        captured_image = None
elif source == 'Test Image':
    captured_image = Image.open("test-images/the-beatles.png")


if captured_image is None:
    st.write("Waiting for image...")
else:
    st.write("Got an image from the {}:".format(source.lower()))

    faces, marked_img = ut.get_faces_from_img(np.array(captured_image))
    st.image(marked_img, use_column_width=True)

    st.subheader('Found {} faces in the picture above:'.format(len(faces)) )

    if(len(faces)):      
        #creating the blank test vector
        data_orig = np.zeros([nr_max_faces, 48,48])

        nr_faces = min(len(faces), 20)

        #putting face data into the vector (only first few), max 20 faces
        for i in range(0, nr_faces):
            data_orig[i,:,:] = ut.contrast_stretch(faces[i,:,:])

            #preparing image and putting it into the batch 
            n = data_orig.shape[0]
            data = np.zeros([n,48**2])
            for i in range(n):
                xx = data_orig[i,:,:]
                xx -= np.mean(xx)
                xx /= np.linalg.norm(xx)
                data[i,:] = xx.reshape(2304); #np.reshape(xx,[-1])

        result = sess.run([y], feed_dict={xin: data, keep_prob_input: 1.0})
        
        for i in range(0, nr_faces):
            emotion_nr = np.argmax(result[0][i])

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))
            ax1.imshow(np.reshape(data[i,:], (48,48)))
            ax1.axis('off')
            ax1.set_title(str_emotions[emotion_nr])
            ax2.bar(np.arange(nc) , result[0][i])
            ax2.set_xticks(np.arange(nc))
            ax2.set_xticklabels(str_emotions, rotation=45)
            ax2.set_yticks([])

            st.pyplot(plt)