# ONNX Models (Offline)

This folder contains ONNX models that are vendored into the repo so the app can run offline and E2E tests can validate features reliably.

## Included models

- `arcface.onnx`
  - Task: face embedding (recognition)
  - Source: InsightFace model `w600k_r50.onnx` (as redistributed by `yakhyo/facial-analysis`)
  - SHA256: `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43`

- `emotion-ferplus-8.onnx`
  - Task: emotion classification (8 classes)
  - SHA256: `a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c`

- `gender_googlenet.onnx`
  - Task: gender *appearance* classification (2 classes)
  - SHA256: `af24a4eaa9eaf70913cc9a337a0387c86f11549cbd9bbc16bffeefcdcf88cbf4`

- `age_googlenet.onnx`
  - Task: age bucket classification (8 buckets)
  - SHA256: `fa2a3228e425056aa2b080b3afd3cf607327c86616e952602ed67b5fc16ab356`

- `genderage.onnx`
  - Task: gender + age attributes (single inference, 2-in-1)
  - Source: `yakhyo/facial-analysis` (as redistributed by PhotoPrism mirror)
  - SHA256: `4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb`

- `silero_vad.onnx`
  - Task: voice activity detection (speech probability)
  - Source: `snakers4/silero-vad` (MIT)
  - SHA256: `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`

- `face_mesh_Nx3x192x192_post.onnx`
  - Task: dense face landmarks (FaceMesh) for mouth metrics
  - Source: `PINTO0309/facemesh_onnx_tensorrt` (Apache-2.0)
  - SHA256: `ae6ada9004f22be3ab6bab8cbfbc8b795f8595f69e5eea77b95cb2fc344c04eb`

## Licenses and attribution

See:
- `THIRD_PARTY_NOTICES.md`
- `LICENSE.onnx-models-Apache-2.0.txt`
- `LICENSE.litehub-MIT.txt`
- `LICENSE.silero-vad-MIT.txt`
- `LICENSE.facemesh_onnx_tensorrt-Apache-2.0.txt`
