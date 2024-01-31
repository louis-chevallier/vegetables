FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

ENV TZ=Europe/Paris
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV DEBIAN_FRONTEND=noninteractive
RUN rm --force /etc/apt/sources.list.d/cuda.list && rm --force /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update --fix-missing
RUN apt-get install --assume-yes --no-install-recommends  build-essential make git gzip tar wget cmake libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Get source code project
WORKDIR /code
COPY . /code/cara/smoke/smoke/face_fit

#RUN git clone -b qarnot https://jitdeep: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@github.com/jitdeep/cara.git
COPY requirements.txt  /code/cara/smoke/smoke/face_fit

# Setup environment
WORKDIR /code/cara/smoke/smoke/face_fit
RUN bash -ic 'source buildenv.sh && build_python_env_docker install'

# Set Path to use Python
ENV PATH="$PATH:/code/cara/smoke/smoke/face_fit/conda_docker/bin"

# Get pandas
#RUN pip install pandas

RUN bash -ic 'source buildenv.sh && testtorchf  && make test_docker'
# Get data from gdrive
WORKDIR /data/runtime_data
COPY  runtime_data .

WORKDIR /code/cara/smoke/smoke/face_fit
