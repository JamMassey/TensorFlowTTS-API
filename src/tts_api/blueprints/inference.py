from __future__ import annotations

import logging

from flask import Blueprint

logger = logging.getLogger(__name__)


inference_blueprint = Blueprint("tasks", __name__)
