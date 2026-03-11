import argparse
import os
import sys

import torch
import torch.nn as nn


def _load_state(path: str):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


class TalkNetAsdWrapper(nn.Module):
    def __init__(self, talknet_repo_dir: str):
        super().__init__()
        sys.path.insert(0, talknet_repo_dir)
        from model.talkNetModel import talkNetModel  # pylint: disable=import-error
        from loss import lossAV  # pylint: disable=import-error

        self.backbone = talkNetModel()
        self.head = lossAV()

    def forward(self, audio_mfcc, visual_gray):
        # audio_mfcc: [B, Ta, 13]
        # visual_gray: [B, Tv, 112, 112]
        a = self.backbone.forward_audio_frontend(audio_mfcc)
        v = self.backbone.forward_visual_frontend(visual_gray)
        a, v = self.backbone.forward_cross_attention(a, v)
        x = self.backbone.forward_audio_visual_backend(a, v)
        logits = self.head.FC(x)
        probs = torch.softmax(logits, dim=-1)
        # Return speaking prob (class 1) as [B*Tv]
        return probs[:, 1]


def _load_into(wrapper: TalkNetAsdWrapper, state: dict):
    mapped = {}
    for k, v in state.items():
        # talkNet wrapper prefixes:
        #  - model.* -> backbone.*
        #  - lossAV.FC.* -> head.FC.*
        if k.startswith("model."):
            mapped["backbone." + k[len("model."):]] = v
        elif k.startswith("lossAV.FC."):
            mapped["head.FC." + k[len("lossAV.FC."):]] = v
    missing, unexpected = wrapper.load_state_dict(mapped, strict=False)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--talknet_repo", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    wrapper = TalkNetAsdWrapper(args.talknet_repo)
    wrapper.eval()

    state = _load_state(args.weights)
    _load_into(wrapper, state)

    # Dummy inputs:
    B = 1
    Tv = 25  # 1 second @25fps
    Ta = Tv * 4  # MFCC frames
    audio = torch.randn(B, Ta, 13, dtype=torch.float32)
    visual = torch.randint(low=0, high=255, size=(B, Tv, 112, 112), dtype=torch.float32)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (audio, visual),
        args.out,
        opset_version=17,
        input_names=["audio_mfcc", "visual_gray"],
        output_names=["speaking_prob"],
        dynamic_axes={
            "audio_mfcc": {0: "batch", 1: "audio_frames"},
            "visual_gray": {0: "batch", 1: "video_frames"},
            "speaking_prob": {0: "flat_time"},
        },
    )
    print("ONNX exported to:", args.out)


if __name__ == "__main__":
    main()

