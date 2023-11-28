FROM tensorflow/tensorflow:2.7.0-gpu
ENV PYTHONUNBUFFERED=1

WORKDIR /api
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update
RUN apt-get install -y zsh tmux wget git libsndfile1
RUN python -m pip install --upgrade pip

COPY requirements.txt /api/
RUN pip install -r requirements.txt

RUN pip install ipython && \
    pip install git+https://github.com/TensorSpeech/TensorflowTTS.git && \
    pip install git+https://github.com/repodiac/german_transliterate.git#egg=german_transliterate

COPY setup.py /api/setup.py
COPY setup.cfg /api/setup.cfg
COPY versioneer.py /api/versioneer.py
COPY pyproject.toml /api/pyproject.toml
COPY README.md /api/README.md
COPY src /api/src

RUN pip install .

EXPOSE 5000
ENTRYPOINT [ "python3 src/tts_api/__main__.py" ]
