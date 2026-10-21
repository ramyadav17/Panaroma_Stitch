from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from panorama import resize_for_processing, stitch_images

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 45 * 1024 * 1024


@app.get("/")
def index():
    return render_template("index.html")


def read_upload(upload) -> np.ndarray:
    data = np.frombuffer(upload.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{upload.filename} is not a supported image")
    return resize_for_processing(image)


@app.post("/api/stitch")
def stitch():
    try:
        uploads = request.files.getlist("images")
        if len(uploads) < 2:
            return jsonify(error="Please select at least two images."), 400
        method = request.form.get("method", "sift").lower()
        blend = request.form.get("blend", "feather").lower()
        if method not in {"sift", "orb"} or blend not in {"feather", "hard"}:
            return jsonify(error="Invalid stitching settings."), 400
        images = [read_upload(upload) for upload in uploads]
        panorama, diagnostics = stitch_images(images, method=method, blend=blend,
                                              ratio=float(request.form.get("ratio", 0.75)),
                                              ransac_threshold=float(request.form.get("threshold", 5.0)))
        ok, encoded = cv2.imencode(".jpg", panorama, [cv2.IMWRITE_JPEG_QUALITY, 94])
        if not ok:
            raise RuntimeError("Could not encode panorama")
        return jsonify(image="data:image/jpeg;base64," + base64.b64encode(encoded).decode(),
                       width=int(panorama.shape[1]), height=int(panorama.shape[0]), diagnostics=diagnostics)
    except (ValueError, RuntimeError, cv2.error, np.linalg.LinAlgError) as error:
        return jsonify(error=str(error)), 400


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error="Uploads are limited to 45 MB in total."), 413


if __name__ == "__main__":
    app.run(debug=True)
