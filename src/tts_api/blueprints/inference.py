from __future__ import annotations

import logging
import os

from flask import Blueprint, abort, current_app, jsonify, request, send_file

from tts_api.utils.file_utils import temp_zip

logger = logging.getLogger(__name__)


inference_blueprint = Blueprint("tasks", __name__)


@inference_blueprint.route("/upload_model", methods=["POST"])
def upload_model():
    """Upload a model to the filesystem."""

    if "model" not in request.files:
        abort(400, "No model file provided.")
    if "type" not in request.form:
        abort(400, "No model type provided.")

    type = request.form["type"]
    filename = request.form.get("filename", request.files["model"].filename)
    filesystem_root = current_app.config["FILESYSTEM_ROOT"]
    if type == "base":
        path = os.path.join(filesystem_root, "models", "base")
    elif type == "vocoder":
        path = os.path.join(filesystem_root, "models", "vocoder")
    else:
        abort(501, "Given model type is not supported yet.")

    request.files["model"].save(os.path.join(path, filename))
    return jsonify({"message": "Model uploaded successfully."})


@inference_blueprint.route("/list_modela", methods=["GET"])
def list_models():
    """List all models in the filesystem."""
    if "type" not in request.form:
        abort(400, "No model type provided.")
    type = request.form["type"]

    filesystem_root = current_app.config["FILESYSTEM_ROOT"]
    if type == "base":
        path = os.path.join(filesystem_root, "models", "base")
    elif type == "vocoder":
        path = os.path.join(filesystem_root, "models", "vocoder")
    else:
        abort(501, "Given model type is not supported yet.")

    models = os.listdir(path)
    return jsonify({"models": models})


@inference_blueprint.route("/delete_model", methods=["DELETE"])
def delete_model():
    """Delete a model from the filesystem."""
    if "type" not in request.form:
        abort(400, "No model type provided.")
    if "filename" not in request.form:
        abort(400, "No model filename provided.")

    type = request.form["type"]
    filename = request.form["filename"]
    filesystem_root = current_app.config["FILESYSTEM_ROOT"]
    if type == "base":
        path = os.path.join(filesystem_root, "models", "base")
    elif type == "vocoder":
        path = os.path.join(filesystem_root, "models", "vocoder")
    else:
        abort(501, "Given model type is not supported yet.")

    os.remove(os.path.join(path, filename))
    return jsonify({"message": "Model deleted successfully."})


@inference_blueprint.route("/download_model", methods=["GET"])
def download_model():
    """Download a model from the filesystem."""
    if "type" not in request.form:
        abort(400, "No model type provided.")
    if "filename" not in request.form:
        abort(400, "No model filename provided.")

    type = request.form["type"]
    filename = request.form["filename"]
    filesystem_root = current_app.config["FILESYSTEM_ROOT"]
    if type == "base":
        path = os.path.join(filesystem_root, "models", "base")
    elif type == "vocoder":
        path = os.path.join(filesystem_root, "models", "vocoder")
    else:
        abort(501, "Given model type is not supported yet.")
    path = os.path.join(path, filename)
    zip_path = temp_zip(path)
    response = send_file(zip_path, as_attachment=True, attachment_filename=f"{filename}.zip")
    try:
        os.unlink(zip_path)
    except Exception as e:
        logger.warning(f"Failed to delete temp zip file: {e}")
    return response
