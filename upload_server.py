import os
import uuid
from pathlib import Path

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

UPLOAD_ROOT = Path(os.environ.get("PV_UPLOAD_DIR", "/mnt/DATA/proteoviewer_uploads"))
MAX_UPLOAD_MB = int(os.environ.get("PV_MAX_UPLOAD_MB", "5000"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


@app.post("/upload")
def upload():
    if "file" not in request.files:
        return jsonify(ok=False, error="No file field in request."), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify(ok=False, error="No filename provided."), 400

    filename = secure_filename(f.filename)
    if not filename.endswith(".h5ad"):
        return jsonify(ok=False, error="Only .h5ad files are accepted."), 400

    upload_id = uuid.uuid4().hex
    dest_dir = UPLOAD_ROOT / upload_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_path = dest_dir / filename
    f.save(dest_path)

    return jsonify(
        ok=True,
        upload_id=upload_id,
        filename=filename,
        path=str(dest_path),
        size_bytes=dest_path.stat().st_size,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PV_UPLOAD_PORT", "5010"))
    app.run(host="127.0.0.1", port=port)
