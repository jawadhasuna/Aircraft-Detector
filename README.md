# Aircraft Detector

### 🔗 [https://jawadhassanbusiness--sar-detector.modal.run](https://jawadhassanbusiness--sar-detector.modal.run)

\---

Aircraft Detector identifies commercial aircraft in Synthetic Aperture Radar (SAR) imagery. SAR is a radar imaging technique that produces images through cloud, smoke, and darkness — conditions where optical cameras fail — which makes automated interpretation of it valuable but difficult, since SAR images look nothing like ordinary photographs.

The model is a **YOLOv8n** network fine-tuned on the **SAR-ACD** dataset across six airframe classes: A220, A320/321, A330, ARJ21, Boeing 737, and Boeing 787. After training it was quantized to **INT8 ONNX** using dynamic weight-only quantization, reducing the model to **3.3 MB** and allowing it to run on CPU alone with no GPU required.

Inference runs on **onnxruntime 1.24.4** with **NumPy** and **Pillow** handling image preprocessing, wrapped in a **FastAPI** service. The application is deployed on **Modal**, a serverless cloud platform, where it scales to zero when idle and starts on demand. The same endpoint serves both the API and the browser interface, so the site is publicly available at all times without a dedicated server.

Upload a SAR image through the web interface and the model returns the aircraft class with a confidence score, drawn over the image.

\---

**Note on output:** SAR-ACD is a classification dataset — each training image was assigned a single bounding box covering the full frame. The model therefore identifies *which* aircraft is present rather than *where* it sits, and returns one whole-image box. True localization would require per-aircraft annotations and retraining.

**Runtime requirement:** onnxruntime **1.24 or newer** is mandatory. Versions 1.20–1.23 and all native `onnxruntime-node` builds fail on this model's `ConvInteger` operators with `NOT\_IMPLEMENTED`.

\---

## License

Released under **AGPL-3.0**, inherited from Ultralytics YOLOv8, on which the model is based.

