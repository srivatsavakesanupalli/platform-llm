FROM cibiai.azurecr.io/gpu_cu116py39:latest
WORKDIR /app
ENV DEBIAN_FRONTEND noninteractive
RUN apt update -y && apt install git ffmpeg unzip curl libc-dev libpq-dev libsm6 libopenblas-dev libxext6 make automake gcc g++ subversion python3-dev wget -y
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt
ADD configs configs
ADD deployer deployer
ADD exceptions exceptions
ADD reader reader
ADD repo repo
ADD storage storage
ADD models models
ADD trainer trainer
ADD templates templates
ADD samples samples
COPY app.py app.py
COPY auth.py auth.py
COPY db.py db.py
COPY utils.py utils.py
COPY train.py train.py
EXPOSE 80
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
RUN ln -f -s /usr/local/lib/libmkl_intel_lp64.so.2 /usr/local/lib/libmkl_intel_lp64.so
RUN ln -f -s /usr/local/lib/libmkl_sequential.so.2 /usr/local/lib/libmkl_sequential.so  
RUN ln -s -f /usr/local/lib/libmkl_core.so.2 /usr/local/lib/libmkl_core.so
CMD uvicorn app:app --port 80 --host 0.0.0.0
