# Face Emotionz Web-App 

Streamlit app using OpenCV to recognize faces in a photo and my custom Tensorflow DNN to recognize emotions in those faces. 

![demo.gif](test-images/demo.gif)

# Deployment

The app is deployed in the free-tier streamlit sharing, [try it here](https://share.streamlit.io/patryk-oleniuk/face-emotionz-webapp/main/app/streamlit_app.py).

# Dev

```
streamlit run app/streamlit_app.py --server.maxUploadSize=5
```

# Requirements
- tensorflow
- OpenCV