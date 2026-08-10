"""
SAR Aircraft Detector — deployed on Modal.

Deploy with:   modal deploy modal_app.py
Test locally:  modal serve modal_app.py
"""

from pathlib import Path

import modal

APP_NAME = "sar-aircraft-detector"

HERE = Path(__file__).parent
MODEL_REMOTE_PATH = "/model/yolov8n_int8.onnx"
ASSETS_REMOTE_PATH = "/assets"

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
# onnxruntime (the Python build) is what actually runs this INT8 model well —
# the Node native runtime is missing the ConvInteger kernel this model needs.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        # 1.24.x is the first release that can run this model's ConvInteger
        # nodes on CPU. 1.20 / 1.22 / 1.23 all fail with NOT_IMPLEMENTED.
        # Do not downgrade this pin.
        "onnxruntime==1.24.4",
        "numpy==2.5.2",
        "pillow==12.3.0",
        "fastapi[standard]==0.115.4",
        "python-multipart==0.0.12",
    )
    # copy=True bakes these into the image layer so the container starts
    # with everything already present (faster, more predictable cold starts)
    .add_local_file(HERE / "models" / "yolov8n_int8.onnx", MODEL_REMOTE_PATH, copy=True)
    .add_local_dir(HERE / "frontend", ASSETS_REMOTE_PATH, copy=True)
    .add_local_python_source("detector")
)

app = modal.App(APP_NAME, image=image)


@app.function(
    # keep the container warm for 5 min after the last request, so repeat
    # visitors don't pay the cold-start cost; after that it scales to zero
    # and costs nothing
    scaledown_window=300,
    # never hold a container open when idle — this is what keeps the bill at $0
    min_containers=0,
    max_containers=2,
    timeout=60,
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app(label="sar-detector")
def web():
    import onnxruntime as ort
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles

    from detector import CLASS_NAMES, run_detection

    # This function body runs once per container start, so the model is
    # loaded a single time and reused across every request that container
    # handles — not reloaded per request.
    session = ort.InferenceSession(
        MODEL_REMOTE_PATH, providers=["CPUExecutionProvider"]
    )
    print("ONNX model loaded:", [i.name for i in session.get_inputs()])

    web_app = FastAPI(title="SAR Aircraft Detector")

    @web_app.post("/detect")
    async def detect(image: UploadFile = File(...)):
        try:
            data = await image.read()
            if not data:
                return JSONResponse({"error": "Empty file."}, status_code=400)

            detections, meta = run_detection(session, data)

            return {
                "detections": detections,
                "imageWidth": meta["orig_w"],
                "imageHeight": meta["orig_h"],
            }
        except Exception as exc:
            print("Inference error:", repr(exc))
            return JSONResponse(
                {"error": f"Inference failed: {exc}"}, status_code=500
            )

    @web_app.get("/health")
    def health():
        return {"status": "ready", "classes": CLASS_NAMES}

    # Serve index.html / style.css / app.js at the root path.
    # Mounted last so it doesn't shadow the API routes above.
    web_app.mount("/", StaticFiles(directory=ASSETS_REMOTE_PATH, html=True))

    return web_app