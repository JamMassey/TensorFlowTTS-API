from __future__ import annotations

import logging
import os

import gdown
import tensorflow as tf
from tensorflow_tts.inference import AutoConfig, AutoProcessor, TFAutoModel

logger = logging.getLogger(__name__)


try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    logger.warning("Did not import matplotlib. Please ensure to install the right extras package if this was unintentional.")


TENSORFLOW_TTS_TEXT2MEL_PATHS = {
    "TACOTRON": "tensorspeech/tts-tacotron2-ljspeech-en",
    "FASTSPEECH": "tensorspeech/tts-fastspeech-ljspeech-en",
    "FASTSPEECH2": "tensorspeech/tts-fastspeech2-ljspeech-en",
}

TENSORFLOW_TTS_VOCODER_PATHS = {
    "MELGAN": "tensorspeech/tts-melgan-ljspeech-en",
    # "MELGAN-STFT": "tensorspeech/tts-melgan-stft-ljspeech-en",
    "MB-MELGAN": "tensorspeech/tts-mb_melgan-ljspeech-en",
}


def tacotron2_inference(
    text2mel_model: TFAutoModel, input_ids: tf.Tensor, speaker_ids: tf.Tensor = tf.convert_to_tensor([0], dtype=tf.int32)
):
    decoder_outputs, mel_outputs, stop_token_predictions, alignment_history = text2mel_model.inference(
        tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0), tf.convert_to_tensor([len(input_ids)], tf.int32), speaker_ids
    )
    return mel_outputs


def fastspeech_inference(
    text2mel_model: TFAutoModel,
    input_ids: tf.Tensor,
    speaker_ids: tf.Tensor = tf.convert_to_tensor([0], dtype=tf.int32),
    speed_ratios: tf.Tensor = tf.convert_to_tensor([1.0], dtype=tf.float32),
):
    mel_before, mel_outputs, duration_outputs = text2mel_model.inference(
        tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0), speaker_ids, speed_ratios
    )
    return mel_outputs


def fastspeech2_inference(
    text2mel_model: TFAutoModel,
    input_ids: tf.Tensor,
    speaker_ids: tf.Tensor = tf.convert_to_tensor([0], dtype=tf.int32),
    speed_ratios: tf.Tensor = tf.convert_to_tensor([1.0], dtype=tf.float32),
    f0_ratios: tf.Tensor = tf.convert_to_tensor([1.0], dtype=tf.float32),
    energy_ratios: tf.Tensor = tf.convert_to_tensor([1.0], dtype=tf.float32),
):
    mel_before, mel_outputs, duration_outputs, f0_outputs, energy_outputs = text2mel_model.inference(
        tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0), speaker_ids, speed_ratios, f0_ratios, energy_ratios
    )
    return mel_outputs


TENSORFLOW_TTS_FUNCTIONS = {
    "TACOTRON": tacotron2_inference,
    "FASTSPEECH": fastspeech_inference,
    "FASTSPEECH2": fastspeech2_inference,
}


def load_custom_model(model_name: str, model_path: str):
    config = AutoConfig.from_pretrained(model_path)
    model = TFAutoModel.from_pretrained(config=config, pretrained_path=model_path, name=model_name)
    return model


def download_melgan_sft(output_folder):
    gdown.download(
        "https://drive.google.com/uc?id=1WB5iQbk9qB-Y-wO8BU6S2TnRiu4VU5ys", os.path.join(output_folder, f"melgan_stft.yml"), quiet=False
    )
    gdown.download(
        "https://drive.google.com/uc?id=1OqdrcHJvtXwNasEZP7KXZwtGUDXMKNkg", os.path.join(output_folder, f"melgan_stft.h5"), quiet=False
    )


# For now, if you want to register MelGAN-STFT, you need to do so manually using the functions provided.
class Synthesizer:
    def __init__(self):
        self.text2mel_models = {}
        self.text2mel_model_functions = {}
        self.vocoder_models = {}
        self.processors = {}

    def load_text2mel_model(self, model_name: str, model: TFAutoModel, model_function):
        self.text2mel_models[model_name] = model
        self.text2mel_model_functions[model_name] = model_function

    def load_vocoder_model(self, model_name: str, model: TFAutoModel):
        self.vocoder_models[model_name] = model

    def load_processor(self, processor_name: str, processor: AutoProcessor):
        self.processors[processor_name] = processor

    def register_known_text2mel_model(self, model_name: str):
        self.load_text2mel_model(
            model_name,
            TFAutoModel.from_pretrained(TENSORFLOW_TTS_TEXT2MEL_PATHS[model_name], name=model_name),
            TENSORFLOW_TTS_FUNCTIONS[model_name],
        )

    def register_known_vocoder_model(self, model_name: str):
        self.load_vocoder_model(model_name, TFAutoModel.from_pretrained(TENSORFLOW_TTS_VOCODER_PATHS[model_name], name=model_name))

    def register_known_processor(self, model_name: str):
        self.load_processor(model_name, AutoProcessor.from_pretrained(TENSORFLOW_TTS_TEXT2MEL_PATHS[model_name]))

    def register_known_models(self):
        for model_name, model_path in TENSORFLOW_TTS_TEXT2MEL_PATHS.items():
            self.register_known_text2mel_model(model_name)
            self.register_known_processor(model_name)
        for model_name, model_path in TENSORFLOW_TTS_VOCODER_PATHS.items():
            self.register_known_vocoder_model(model_name)

    def list_loaded_text2mel_models(self):
        return list(self.text2mel_models.keys())

    def list_loaded_vocoders(self):
        return list(self.vocoder_models.keys())

    def list_loaded_processors(self):
        return list(self.processors.keys())

    def list_loaded_models(self):
        return {"text2mel": self.list_text2mel_models(), "vocoder": self.list_vocoder_models(), "processor": self.list_processors()}

    def list_tensorflow_tts_models(self):
        return {"text2mel": TENSORFLOW_TTS_TEXT2MEL_PATHS.keys(), "vocoder": TENSORFLOW_TTS_VOCODER_PATHS.keys()}

    def do_synthesis(self, input_text: str, text2mel_name: str, vocoder_name: str, processor_name: str | None = None):
        input_ids = self.processors[processor_name if processor_name is not None else text2mel_name].text_to_sequence(input_text)
        mel_outputs = self.text2mel_model_functions[text2mel_name](self.text2mel_models[text2mel_name], input_ids)
        audio = self.vocoder_models[vocoder_name](mel_outputs)[0, :, 0]
        return mel_outputs.numpy(), audio.numpy()


def visualize_mel_spectrogram(mels: np.ndarray):
    mels = tf.reshape(mels, [-1, 80]).numpy()
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(311)
    ax1.set_title(f"Predicted Mel-after-Spectrogram")
    im = ax1.imshow(np.rot90(mels), aspect="auto", interpolation="none")
    fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax1)
    plt.show()
