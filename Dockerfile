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

RUN rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.10 python3.10-dev python3.10-distutils
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --set python3 /usr/bin/python3.10

RUN useradd -m -u 1000 user

RUN mkdir -p /cache/hub /cache/transformers /cache/torch && \
    chmod 777 /cache && \
    chmod 777 /cache/hub && \
    chmod 777 /cache/transformers && \
    chmod 777 /cache/torch

RUN mkdir -p /home/user/.cache/huggingface
RUN chown -R user:user /home/user
RUN chmod -R 755 /home/user

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
ENV TRANSFORMERS_CACHE=/cache/transformers
ENV TORCH_HOME=/cache/torch
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6"
ENV TCNN_CUDA_ARCHITECTURES=86;80;75;70;61;60
ENV FORCE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH}
ENV CMAKE_PREFIX_PATH="/home/user/.local/lib/python3.10/site-packages/torch"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel ninja --timeout 60 --retries 2
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir onnxruntime-gpu --timeout 300 --retries 5
RUN pip install huggingface-hub
RUN pip install boto3 diffusers torch
RUN pip install ollama beautifulsoup4 pydantic instructor openai
RUN pip install eval-type-backport
RUN pip install bitsandbytes
RUN pip install --no-cache-dir accelerate
RUN pip install --no-cache-dir --upgrade transformers diffusers
RUN pip install --no-cache-dir sentencepiece
RUN pip install --no-cache-dir optimum-quanto
RUN pip install --no-cache-dir optimum

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

RUN pip install --upgrade diffusers huggingface-hub transformers torchvision
RUN pip install xatlas

ENTRYPOINT ["/home/user/entrypoint.sh"]
