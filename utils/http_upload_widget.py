import param
import panel as pn


class HttpUploadWidget(pn.reactive.ReactiveHTML):
    upload_url = param.String(default="/upload")
    accept = param.String(default=".h5ad")

    upload_path = param.String(default="")
    filename = param.String(default="")
    message = param.String(default="")
    progress = param.Integer(default=0)
    busy = param.Boolean(default=False)

    _template = """
    <div style="display:flex; align-items:center; gap:10px; width:100%;">
      <label
        for="file_input"
        style="
          background:#0d6efd;
          color:white;
          padding:6px 12px;
          border-radius:4px;
          cursor:pointer;
          font-size:13px;
          font-weight:500;
          white-space:nowrap;
        "
      >
        Browse .h5ad file
      </label>

      <input
        id="file_input"
        type="file"
        accept="${accept}"
        onchange="${script('upload')}"
        style="display:none;"
      />

      <progress
        id="progress_bar"
        value="${progress}"
        max="100"
        style="width:220px; display:${busy ? 'inline-block' : 'none'};"
      ></progress>

      <span id="message" style="font-size:13px;">${message}</span>
    </div>
    """

    _scripts = {
        "upload": """
            const file = file_input.files[0];

            if (!file) {
                data.message = "";
                return;
            }

            if (!file.name.endsWith(".h5ad")) {
                data.message = "Only .h5ad files are accepted.";
                return;
            }

            data.busy = true;
            data.progress = 0;
            data.upload_path = "";
            data.filename = "";
            data.message = "Uploading " + file.name + "...";

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
                data.busy = false;

                if (xhr.status === 200) {
                    const resp = JSON.parse(xhr.responseText);

                    if (resp.ok) {
                        data.upload_path = resp.path;
                        data.filename = resp.filename;
                        data.progress = 100;
                        data.message = "";
                    } else {
                        data.message = "Upload failed: " + (resp.error || "unknown error");
                    }
                } else {
                    data.message = "Upload failed: HTTP " + xhr.status;
                }

                file_input.value = "";
            };

            xhr.onerror = function() {
                data.busy = false;
                data.message = "Upload failed.";
                file_input.value = "";
            };

            xhr.send(formData);
        """
    }
