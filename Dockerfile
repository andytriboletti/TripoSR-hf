FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ARG USER_NAME=user
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update --option Acquire::Retries=5

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libglib2.0-0

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libsm6 \
    libxext6 \
    libxrender1

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python-is-python3 \
    python3-dev \
    python3-pip

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    wget \
    tmux \
    nano

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libopencv-dev \
    python3-opencv \
    libglib2.0-0

RUN rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y software-properties-common
#RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.10 python3.10-dev python3.10-distutils
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --set python3 /usr/bin/python3.10

RUN apt-get update && apt-get install -y openssh-client

    # Install CUDNN Advanced package
    RUN apt-get update && apt-get install -y \
        libcudnn9-cuda-12 \
        libcudnn9-dev-cuda-12

  # Update library cache
  RUN ldconfig

  # Install Intel TBB 2021 update 6 or later
RUN apt-get update && apt-get install -y \
libtbb-dev \
libtbb12

# Set TBB environment variable
ENV TBB_INTERFACE_VERSION=12060


  # Set environment variables for CUDNN
  ENV CUDNN_PATH="/usr/lib/x86_64-linux-gnu/libcudnn.so.9"
  ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

RUN useradd -m -u 1000 user

RUN mkdir -p /cache/hub /cache/transformers /cache/torch && \
    chmod 777 /cache && \
    chmod 777 /cache/hub && \
    chmod 777 /cache/transformers && \
    chmod 777 /cache/torch

RUN mkdir -p /home/user/.cache/huggingface
RUN chown -R user:user /home/user
RUN chmod -R 755 /home/user

# Install TensorRT packages
RUN apt-get update && apt-get install -y \
    libnvinfer8 \
    libnvinfer-dev \
    libnvinfer-plugin8 \
    python3-libnvinfer \
    python3-libnvinfer-dev

# Update library cache
RUN ldconfig

# Set TensorRT library path
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/tensorrt:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

USER user

ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH
ENV PYTHONPATH=$HOME/app
ENV PYTHONUNBUFFERED=1
ENV GRADIO_ALLOW_FLAGGING=never
ENV GRADIO_NUM_PORTS=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_THEME=huggingface
ENV SYSTEM=spaces
ENV HF_HOME=/cache
ENV HUGGING_FACE_HUB_CACHE=/cache/hub
# ENV TRANSFORMERS_CACHE=/cache/transformers
ENV TORCH_HOME=/cache/torch
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6"
ENV TCNN_CUDA_ARCHITECTURES=86;80;75;70;61;60
ENV FORCE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH}
ENV CMAKE_PREFIX_PATH="/home/user/.local/lib/python3.10/site-packages/torch"

ENV PIP_DEFAULT_TIMEOUT=100

RUN python -m pip install --upgrade pip --timeout=100

RUN pip install --no-cache-dir --upgrade pip setuptools wheel ninja --timeout 60 --retries 2



RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  --timeout=100


RUN pip install --no-cache-dir \
    onnxruntime-gpu \
    huggingface-hub \
    boto3 \
    diffusers \
    torch \
    ollama \
    beautifulsoup4 \
    pydantic \
    openai

RUN pip install eval-type-backport --upgrade
RUN pip install bitsandbytes --upgrade
RUN pip install --no-cache-dir accelerate --upgrade
RUN pip install --no-cache-dir --upgrade transformers diffusers


RUN pip install --no-cache-dir sentencepiece --upgrade
RUN pip install --no-cache-dir optimum-quanto --upgrade
RUN pip install --no-cache-dir optimum --upgrade



# Install TensorRT Python packages
RUN pip install tensorrt



RUN python -c "import torch; print(torch.version.cuda)"
RUN python -c "import torch; CMAKE_PREFIX_PATH=torch.utils.cmake_prefix_path" && \
    export CMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')"

COPY requirements.txt /tmp
RUN cd /tmp && \
    grep -v torchmcubes requirements.txt > requirements_filtered.txt && \
    pip install --no-cache-dir -r requirements_filtered.txt && \
    git clone https://github.com/tatsy/torchmcubes.git && \
    cd torchmcubes && \
    pip install .

VOLUME ["/mnt/c", "/cache"]

RUN echo '#!/bin/bash\nchmod -R 777 /cache\nexec "$@"' > /home/user/entrypoint.sh
RUN chmod +x /home/user/entrypoint.sh

RUN pip install --upgrade diffusers huggingface-hub

RUN pip install --upgrade transformers torchvision

RUN pip install xatlas
RUN pip install --no-cache-dir instructor --upgrade
RUN pip install xformers

RUN pip install --no-cache-dir \
    rq \
    redis


RUN pip install --no-cache-dir \
    moderngl \
    moderngl-window

# Install rembg and other required packages
RUN pip install --no-cache-dir \
    rembg \
    opencv-python-headless \
    Pillow \
    requests

# Install ONNX Runtime GPU
#RUN pip uninstall -y onnxruntime onnxruntime-gpu && \
#   pip install --no-cache-dir onnxruntime-gpu



COPY monitor_python_jobs.sh /home/user/
COPY run_python_jobs.sh /home/user/
COPY stop_python_jobs.sh /home/user/
#RUN chmod +x /home/user/monitor_and_run_python.sh


# Set TensorRT library path
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/tensorrt:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"


ENTRYPOINT ["/home/user/entrypoint.sh"]

#todo add the following to the dockerfile
#root@andys-pc:/# curl -fsSL https://ollama.com/install.sh | sh
#also have it run ollama run llama3 to download the model
#it should be cachedd in the cache folder
#not really needed for my scripts using llama3 i guess, but for if you want to run it on cmd line


#todo fix the following error
#/home/user/.local/lib/python3.10/site-packages/transformers/utils/hub.py:128: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
#try 1 need to test

#todo fix the following error
#/home/user/.local/lib/python3.10/site-packages/numba/np/ufunc/parallel.py:371: NumbaWarning: The TBB threading layer requires TBB version 2021 update 6 or later i.e., TBB_INTERFACE_VERSION >= 12060. Found TBB_INTERFACE_VERSION = 12050. The TBB threading layer is disabled.  warnings.warn(problem)
#try 1, need to test
#/home/user/.local/lib/python3.10/site-packages/numba/np/ufunc/parallel.py:371: NumbaWarning: The TBB threading layer requires TBB version 2021 update 6 or later i.e., TBB_INTERFACE_VERSION >= 12060. Found TBB_INTERFACE_VERSION = 12050. The TBB threading layer is disabled.
#still error

#todo pip install rq, pip install redis
#try 1, need to test

#todo add
#RUN apt-get update && apt-get install -y openssh-client
#try 1, need to test

#ModuleNotFoundError: No module named 'moderngl'
#try 1, need to test

#todo
#2025-02-19 23:42:22.886590847 [E:onnxruntime:Default, provider_bridge_ort.cc:1848 TryGetProviderInfo_TensorRT] /onnxruntime_src/onnxruntime/core/session/provider_bridge_ort.cc:1539 onnxruntime::Provider& onnxruntime::ProviderLibrary::Get() [ONNXRuntimeError] : 1 : FAIL : Failed to load library libonnxruntime_providers_tensorrt.so with error: libnvinfer.so.10: cannot open shared object file: No such file or directory
#try 1, need to test

#todo ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
#pip install imageio==2.19.3
#scikit-image 0.25.2 requires imageio!=2.35.0,>=2.33, but you have imageio 2.19.3 which is incompatible.
#Successfully installed imageio-2.19.3

#I tried instaling pip install imageio==2.19.3 and got an error:
#ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
#scikit-image 0.25.2 requires imageio!=2.35.0,>=2.33, but you have imageio 2.19.3 which is incompatible.
#pip install imageio-ffmpeg

#todo
#export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0

#todo 
#pip install tbb

#todo
#RUN apt-get update && apt-get install -y xvfb
