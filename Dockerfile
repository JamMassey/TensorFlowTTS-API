FROM tensorflow/tensorflow:2.6.0-gpu
ENV PYTHONUNBUFFERED=1

WORKDIR /api

RUN apt-get install -y zsh tmux wget git libsndfile1
RUN python -m pip install --upgrade pip
RUN pip install ipython && \
    pip install git+https://github.com/TensorSpeech/TensorflowTTS.git && \
    pip install git+https://github.com/repodiac/german_transliterate.git#egg=german_transliterate && 


COPY requirements.txt /api/

RUN pip install -r requirements.txt

COPY setup.py /api/setup.py
COPY setup.cfg /api/setup.cfg
COPY versioneer.py /api/versioneer.py
COPY pyproject.toml /api/pyproject.toml
COPY Resources/ /api/Resources
COPY README.md /api/README.md
COPY src /api/src

EXPOSE 5000
ENTRYPOINT [ "python3"]
CMD ["tts_api/app.py"]
