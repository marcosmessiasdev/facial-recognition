# Third-party notices (models)

The binary model files under `App/models/` are redistributed for local inference.

## Whisper

- GGML model: `models/whisper/ggml-tiny.bin`
- License (code/model): OpenAI Whisper (MIT), see `models/whisper/LICENSE.openai-whisper-MIT.txt`
- The model file was downloaded from a public mirror:
  - https://mirrors.aliyun.com/macports/distfiles/whisper/ggml-tiny.bin

## Notes

- Transcription is probabilistic and may be inaccurate.
- The app should present results as “predictions”.

## TalkNet ASD (weights)

- PyTorch weights: `models/talknet/pretrain_TalkSet.model`
- Source project: `TaoRuijie/TalkNet-ASD` (MIT)
- Download source: Google Drive file referenced by upstream `demoTalkNet.py` (id `1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea`)
- License: `models/talknet/LICENSE.TalkNet-ASD-MIT.txt`
