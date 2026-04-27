import param
import panel as pn


class HttpUploadWidget(pn.reactive.ReactiveHTML):
    upload_url = param.String(default="/upload")
    accept = param.String(default=".h5ad")

    upload_path = param.String(default="")
    filename = param.String(default="")
    message = param.String(default="")
    progress = param.Integer(default=0)
    is_uploading = param.Boolean(default=False)

    _template = """
    <div style="display:flex; align-items:center; gap:10px; width:100%;">
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
          background:#0d6efd;
          color:white;
          border:none;
          border-radius:4px;
          padding:6px 12px;
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
        style="width:220px; display:none;"
      ></progress>

      <span id="message" style="font-size:13px;">${message}</span>
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
                data.message = "Only .h5ad files are accepted.";
                file_input.value = "";
                return;
            }

            data.is_uploading = true;
            data.progress = 0;
            data.upload_path = "";
            data.filename = "";
            data.message = "Uploading " + file.name + "...";

            progress_bar.style.display = "inline-block";
            progress_bar.value = 0;

            const formData = new FormData();
            formData.append("file", file);

            const xhr = new XMLHttpRequest();
            xhr.open("POST", data.upload_url, true);

            xhr.upload.onprogress = function(event) {
                if (event.lengthComputable) {
                    const pct = Math.round((event.loaded / event.total) * 100);
                    data.progress = pct;
                    progress_bar.value = pct;
                }
            };

            xhr.onload = function() {
                if (xhr.status === 200) {
                    const resp = JSON.parse(xhr.responseText);

                    if (resp.ok) {
                        data.progress = 100;
                        progress_bar.value = 100;
                        data.filename = resp.filename;
                        data.upload_path = resp.path;
                        data.message = "";
                    } else {
                        data.message = "Upload failed: " + (resp.error || "unknown error");
                    }
                } else {
                    data.message = "Upload failed: HTTP " + xhr.status;
                }

                data.is_uploading = false;
                progress_bar.style.display = "none";
                file_input.value = "";
            };

            xhr.onerror = function() {
                data.is_uploading = false;
                data.message = "Upload failed.";
                progress_bar.style.display = "none";
                file_input.value = "";
            };

            xhr.send(formData);
        """
    }
