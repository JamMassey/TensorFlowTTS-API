from __future__ import annotations

import logging

from flask import Flask

from tts_api.blueprints import blueprints
from tts_api.utils.args_utils import parse_flask_server_args
from tts_api.utils.logging_utils import setup_logger

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SESSION_TYPE"] = "filesystem"
for blueprint in blueprints:
    app.register_blueprint(blueprint)


@app.route("/healthcheck")
def healthcheck():
    return "OK"


def main(host: str = "0.0.0.0", port: int = 5000, debug: bool = False) -> None:
    logger.info("Starting server...")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    args = parse_flask_server_args()
    setup_logger(args.log_level, args.console_log)
    debug = False
    if args.log_level == logging.DEBUG:
        debug = True
    main(args.host, args.port, debug)
