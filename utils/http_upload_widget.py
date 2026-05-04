import param
import panel as pn


class HttpUploadWidget(pn.reactive.ReactiveHTML):
    upload_url = param.String(default="/upload")
    accept = param.String(default=".h5ad")

    upload_path = param.String(default="")
    filename = param.String(default="")
    progress = param.Integer(default=0)
    is_uploading = param.Boolean(default=False)
    progress_class = param.String(default="pv-progress-hidden")
    error = param.String(default="")

    _template = """
    <div style="display:flex; flex-direction:column; align-items:flex-start; gap:6px; width:100%;">
      <style>
        .pv-progress-hidden {
          display: none;
        }
        .pv-progress-visible {
          display: inline-block;
        }
      </style>

      <input
        id="file_input"
        type="file"
        accept="${accept}"
        onchange="${script('upload')}"
        style="display:none;"
      />

      <button
        id="browse_button"
        onclick="${script('choose_file')}"
        style="
          background:#688fd1;
          color:white;
          border:none;
          border-radius:4px;
          padding:6px 12px;
          margin-left: 9px;
          cursor:pointer;
          font-size:13px;
          font-weight:500;
        "
      >
        Browse .h5ad file
      </button>

      <progress
        id="progress_bar"
        value="${progress}"
        max="100"
        class="${progress_class}"
        style="width:260px; margin-left:8px; margin-top:6px;"
      ></progress>

      <span id="error" style="font-size:13px; color:#a33;">${error}</span>
    </div>
    """

    _scripts = {
        "choose_file": """
            file_input.click();
        """,

        "upload": """
            const file = file_input.files[0];

            if (!file) {
                return;
            }

            if (!file.name.endsWith(".h5ad")) {
                data.error = "Only .h5ad files are accepted.";
                file_input.value = "";
                return;
            }

            data.error = "";
            data.progress = 0;
            data.progress_class = "pv-progress-visible";
            data.upload_path = "";
            data.filename = "";
            data.is_uploading = true;

            const formData = new FormData();
            formData.append("file", file);

            const xhr = new XMLHttpRequest();
            xhr.open("POST", data.upload_url, true);

            xhr.upload.onprogress = function(event) {
                if (event.lengthComputable) {
                    data.progress = Math.round((event.loaded / event.total) * 100);
                }
            };

            xhr.onload = function() {
                if (xhr.status === 200) {
                    const resp = JSON.parse(xhr.responseText);

                    if (resp.ok) {
                        data.progress = 100;
                        data.filename = resp.filename;
                        data.upload_path = resp.path;
                    } else {
                        data.error = "Upload failed: " + (resp.error || "unknown error");
                    }
                } else {
                    data.error = "Upload failed: HTTP " + xhr.status;
                }

                data.is_uploading = false;
                data.progress_class = "pv-progress-hidden";
                file_input.value = "";
            };

            xhr.onerror = function() {
                data.error = "Upload failed.";
                data.is_uploading = false;
                data.progress_class = "pv-progress-hidden";
                file_input.value = "";
            };

            xhr.send(formData);
        """
    }
