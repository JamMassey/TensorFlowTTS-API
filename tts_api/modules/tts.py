
import logging
import tempfile
import jsonify
import tensorflow as tf
import tensorflow_tts as tts
from flask import request, send_file, Blueprint, jsonify

logger = logging.getLogger(__name__)

tts_blueprint = Blueprint('tts', __name__)


tts_models = {
    'tensorspeech/tts-mbrola-01-multi-en': {
        'config_path': 'tensorspeech/tts-mbrola-01-multi-en/config.json',
        'checkpoint_path': 'tensorspeech/tts-mbrola-01-multi-en/model.ckpt-1000000',
        'processor_path': 'tensorspeech/tts-mbrola-01-multi-en/processor.json',
    },
    # Add more models here...
}

tts_blueprint = Blueprint('tts', __name__)

@tts_blueprint.route('/tts', methods=['POST'])
def text_to_speech():
    # Get input text and model name from request
    text = request.json['text']
    model_name = request.json.get('model_name', 'tensorspeech/tts-mbrola-01-multi-en')
    
    # Load pre-trained TTS model and processor for the selected model
    if model_name in tts_models:
        tts_config = tts.AutoConfig.from_pretrained(tts_models[model_name]['config_path'])
        tts_model = tts.TFAutoModel.from_pretrained(tts_models[model_name]['checkpoint_path'], config=tts_config)
        tts_processor = tts.AutoProcessor.from_pretrained(tts_models[model_name]['processor_path'])
    else:
        return jsonify(error='Invalid model_name parameter'), 400
    
    # Convert text to input ids
    input_ids = tts_processor.text_to_sequence(text)
    
    # Generate mel spectrogram from input ids
    mel = tts_model.inference(
        input_ids=tf.expand_dims(input_ids, 0),
        speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32)
    )[0]["mel_outputs"]
    
    # Convert mel spectrogram to waveform
    waveform = tts_processor.decode(mel)[0]
    
    # Save waveform to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as wav_file:
        tts.audio.save_wav(wav_file.name, waveform, sample_rate=tts_processor.sampling_rate)
        wav_file.seek(0)
        
        # Return WAV file in Flask response
        return send_file(wav_file, attachment_filename='output.wav', as_attachment=True)
    
# POST /tts HTTP/1.1
# Content-Type: application/json
# EXAMPLE:
# {
#     "text": "Hello, world!",
#     "model_name": "tensorspeech/tts-mbrola-02-multi-en"
# }