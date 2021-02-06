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

import footer

from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

# matplotlib params
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# old tensorflow
tf.disable_eager_execution()

nr_max_faces=20
nc=6

# load my V1 tensorflow graph as default 
y, xin, keep_prob_input = ut.set_tf_model_graph(nr_max_faces)
sess = tf.Session()
saver = tf.train.Saver()

# restore the weights from file
saver.restore(sess, 'mlmodels/face-emotion-recognition/model_6layers.ckpt')

### Streamlit app
st.title("Facial Emotion Recognizer")
st.subheader("&#8592; Choose the image source on the sidebar:")
st.markdown("test image, webcam, or upload")
st.markdown("Frontal face images without glasses work best. Image is not stored or saved in any form.")
st.markdown("Dislaimer: Use this app at your own risk. Result might be mind-boggling.")
st.subheader('''First, OpenCV will detect faces, (based on [this](https://realpython.com/face-recognition-with-python/)).''')
st.subheader('''Then, Tensorflow will recognize their emotions using [my custom neural net](https://github.com/patryk-oleniuk/emotion_recognition).''')

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

    if len(faces):      
        #creating the blank test vector, must be length of batch_size (old v1 tensorflow)
        data_orig = np.zeros([nr_max_faces, 48,48])

        nr_faces = min(len(faces), 20)

        #putting face data into the vector (only first few), max 20 faces
        for i in range(0, nr_faces):
            data_orig[i,:,:] = ut.contrast_stretch(faces[i,:,:])

        #preparing images
        data = ut.preprocess_faces(data_orig)

        # run the DNN
        result = sess.run([y], feed_dict={xin: data, keep_prob_input: 1.0})
        
        for i in range(0, nr_faces):
            with _lock: # https://matplotlib.org/3.3.2/faq/howto_faq.html#working-with-threads
                plt = ut.plot_face(result[0][i], data[i,:])
                st.pyplot(plt)

footer.footer()