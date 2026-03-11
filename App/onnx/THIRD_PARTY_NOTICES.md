# Third-party notices (models)

The ONNX model files under `App/onnx/` are redistributed for local inference.

## Sources

- ONNX Model Zoo repository: `onnx/models` (Apache-2.0)
- Binary mirror used for download: `hd03022163/litehub` (MIT)
  - Used for: `emotion-ferplus-8.onnx`, `gender_googlenet.onnx`, `age_googlenet.onnx`

The face embedding model file `arcface.onnx` is based on the InsightFace model `w600k_r50.onnx`, downloaded from the PhotoPrism public model mirror:

- https://dl.photoprism.app/onnx/models/w600k_r50.onnx
- Mirror notice: https://dl.photoprism.app/onnx/models/NOTICE
- Upstream model pack: https://yakhyo.github.io/facial-analysis/

The combined attributes model file `genderage.onnx` was downloaded from the same model mirror and upstream pack.

The VAD model file `silero_vad.onnx` was downloaded from the official Silero VAD repository:

- https://github.com/snakers4/silero-vad

The FaceMesh landmarks model file `face_mesh_Nx3x192x192_post.onnx` was downloaded from:

- https://github.com/PINTO0309/facemesh_onnx_tensorrt (Apache-2.0)

The speaker embedding model file `nemo_en_titanet_small.onnx` was downloaded from:

- https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models

See the license texts in this folder.

The ASD model file `talknet_asd.onnx` was exported from the TalkNet ASD project weights:

- https://github.com/TaoRuijie/TalkNet-ASD (MIT)
- Weights source: `App/models/talknet/pretrain_TalkSet.model` (Google Drive id referenced by upstream demo)

## Notes

- Outputs are probabilistic and may be inaccurate.
- The app should present results as “predictions”.
