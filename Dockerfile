ARG RASPBIAN_VERSION=stretch
FROM resin/rpi-raspbian:$RASPBIAN_VERSION

ARG RASPBIAN_VERSION
ARG DEBIAN_FRONTEND=noninteractive


# update apt
RUN apt-get update \
&& apt-get install -y --no-install-recommends apt-utils \
# install necessary build tools \
&& apt-get -qy install build-essential cmake pkg-config unzip wget \
# install necessary libraries \
&& apt-get -qy install \
libjpeg-dev \
libtiff5-dev \
libjasper-dev \
libpng12-dev \
libavcodec-dev \
libavformat-dev \
libswscale-dev \
libv4l-dev \
libxvidcore-dev \
libx264-dev \
#Had to break the install into chunks as the deps wouldn't resolve.  \
&& apt-get -qy install \
libgtk2.0-dev \
libgtk-3-dev \
libatlas-base-dev \
gfortran \
python2.7-dev \
python3-dev \
python-pip \
python-numpy \
python3-pip \
python3-numpy \
libraspberrypi0 \
python-setuptools \
python3-setuptools \
# cleanup apt. \
&& apt-get purge -y --auto-remove \
&& rm -rf /var/lib/apt/lists/*

ARG OPENCV_VERSION=4.0.1
ENV OPENCV_VERSION $OPENCV_VERSION

# download latest source & contrib
RUN cd /tmp \
&& wget -c -N -nv -O opencv.zip https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip \
&& unzip opencv.zip \
&& wget -c -N -nv -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip \
&& unzip opencv_contrib.zip \
# build opencv \
&& cd /tmp/opencv-$OPENCV_VERSION \
&& mkdir build \
&& cd build \
&& cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D INSTALL_C_EXAMPLES=ON \
-D BUILD_PYTHON_SUPPORT=ON \
-D BUILD_NEW_PYTHON_SUPPORT=ON \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib-$OPENCV_VERSION/modules \
-D BUILD_EXAMPLES=ON .. \
&& make -j4  \
&& make \
&& make install\
# ldconfig && \
&& make clean \
# cleanup source \
&& cd / \
&& rm -rf /tmp/* \
&& pip install imutils picamera \
&& pip3 install imutils picamera \
        && date \
        && echo "Raspbian $RASPBIAN_VERSION - OpenCV $OPENCV_VERSION Docker Build finished."


ARG VCS_REF
ARG BUILD_DATE
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/sgtwilko/rpi-raspbian-opencv"


ENTRYPOINT ["/bin/bash"]



WORKDIR /app

WORKDIR /app/opt
RUN wget https://dl.google.com/coral/edgetpu_api/edgetpu_api_latest.tar.gz -O edgetpu_api.tar.gz --trust-server-names \
    && tar xzf edgetpu_api.tar.gz \
    && rm edgetpu_api.tar.gz

COPY "./conf/install.sh"  /app/opt/edgetpu_api/
ENTRYPOINT ["install.sh"]

RUN cd /app/opt/edgetpu_api/
ADD ./install.sh /install.sh
RUN chmod +x /install.sh
ENTRYPOINT ./install.sh

COPY "./conf/tpu_models/coco_labels.txt" /app/opt/

#copy supervisord files
COPY "./conf/supervisord.conf" /app/opt/
RUN mkdir /var/log/supervisord/

#supervisord
ENTRYPOINT ["/usr/local/bin/supervisord", "-c", "/app/opt/supervisord.conf"]


WORKDIR /app/opt/edgetpu_api/

RUN apt update
RUN apt install curl gnupg ca-certificates zlib1g-dev libjpeg-dev -y

RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

RUN apt-get install apt-transport-https
RUN apt update

RUN apt install libedgetpu1-std python3 python3-pip python3-edgetpu  -y

#RUN curl https://dl.google.com/coral/python/tflite_runtime-1.14.0-cp37-cp37m-linux_armv7l.whl > tflite_runtime-1.14.0-cp37-cp37m-linux_armv7l.whl
#RUN pip3 install tflite_runtime-1.14.0-cp37-cp37m-linux_armv7l.whl

RUN pip3 install utils
RUN pip3 install numpy
#RUN pip3 install opencv-python
RUN pip3 install multiprocess
RUN pip3 install python-time
RUN pip3 install pillow
RUN pip3 install argparse
RUN pip3 install Flask

RUN apt-get autoremove && apt-get -f install && apt-get update && apt-get upgrade -y
RUN apt-get install -y curl fswebcam
ADD ./webcamcapture.sh /webcamcapture.sh
RUN chmod +x /webcamcapture.sh
ENTRYPOINT ./webcamcapture.sh



#install live camera libraries
RUN apt-get install libgstreamer1.0-0 gstreamer1.0-tools \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly v4l-utils

#install libraries for camera
RUN apt-get install -y --no-install-recommends build-essential wget feh pkg-config libjpeg-dev zlib1g-dev \
    libraspberrypi0 libraspberrypi-dev libraspberrypi-doc libraspberrypi-bin libfreetype6-dev libxml2 libopenjp2-7 \
    libatlas-base-dev libjasper-dev libqtgui4 libqt4-test \
    python3-dev python3-pip python3-setuptools python3-wheel python3-numpy python3-pil python3-matplotlib python3-zmq

WORKDIR /app/opt
WORKDIR /app
COPY app.py /app


RUN apt-get update
RUN apt-get install -qqy x11-apps











EXPOSE 5000

#ENTRYPOINT ["/bin/bash" "-c","/app.py"]

ENTRYPOINT ["/bin/bash"]
#RUN python3 app.py






--
