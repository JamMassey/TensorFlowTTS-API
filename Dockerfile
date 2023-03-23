FROM tensorflow/tensorflow:2.6.0-gpu
RUN apt-get install -y zsh tmux wget git libsndfile1
RUN python -m pip install --upgrade pip
RUN pip install ipython && \
    pip install git+https://github.com/TensorSpeech/TensorflowTTS.git && \
    pip install git+https://github.com/repodiac/german_transliterate.git#egg=german_transliterate && \
    pip install flask
COPY . ./
EXPOSE 5000
ENTRYPOINT [ "python3"]
CMD ["tts_api/app.py"]
