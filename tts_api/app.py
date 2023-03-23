import logging
from flask import Flask
from modules.tts import tts_blueprint
from utility.logging_utils import setup_logger

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.register_blueprint(tts_blueprint)

@app.route('/healthcheck')
def healthcheck():
    return 'OK'

@app.route('/')
def index():
    return 'Hello, world!'

if __name__ == '__main__':
    setup_logger(stream_logs=True)
    logger.info('Starting Flask app')
    app.run(debug=True, host='0.0.0.0', port=5000)

