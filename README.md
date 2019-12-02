# hackathon-parking
computer vision / machine learning parking spot detection and reporting

### this python file is built off
[https://medium.com/@ageitgey/snagging-parking-spaces-with-mask-r-cnn-and-python-955f2231c400]
[https://github.com/matterport/Mask_RCNN]

### to install on windows
install python 3.7.5 64bit (do not install 3.8)
```
pip3 install --upgrade pip
pip3 install numpy
pip3 install opencv-python
pip3 install mrcnn
pip3 install scipy
pip3 install scikit-image
pip3 install keras
pip3 install tensorflow==1.15
```
note: mrcnn/keras requires older version of tensorflow. 1.15 installs both GPU/CPU versions

to run
```
py parking.py
```

### for GPU support
[https://www.tensorflow.org/install/gpu]
update NVidia drivers
install visual studio community  (might not be required - but CUDA install calls for it)
install CUDA toolkit


### another interesting resource is here
[http://cnrpark.it/]
[https://github.com/fabiocarrara/deep-parking]
