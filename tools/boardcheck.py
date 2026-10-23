#!/usr/bin/env python3
"""A733 hardware checks: source strokes, GPU/CPU, dequant and padded decoder.

SR_SRPIPE=/path/srpipe SR_NBG=/path/model.nb SR_MARGIN=64 SR_INPUT_INT16=1 python3 tools/boardcheck.py input.png
Uses the native x2, 352x224 model. Does not modify the running web service.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

binary = os.environ.get("SR_SRPIPE", "/opt/sr/srpipe")
model = os.environ.get("SR_NBG", "/opt/sr/models/animejanai_v3_x2.nb")
margin = os.environ.get("SR_MARGIN", "16")
source = sys.argv[1]


def run(args, **kwargs):
    result = subprocess.run(args, capture_output=True, **kwargs)
    if result.returncode:
        sys.stderr.write(result.stderr.decode(errors="replace"))
    result.check_returncode()
    return result


with tempfile.TemporaryDirectory(prefix="sr-boardcheck-") as temp:
    tmp = Path(temp)
    # Source smaller than a tile, so the checks exercise edge replication too.
    w, h = 320, 180
    gbr = run(["ffmpeg", "-v", "error", "-i", source, "-frames:v", "1", "-vf",
               "scale=320:180", "-pix_fmt", "gbrp", "-f", "rawvideo", "-"]).stdout
    nv = run(["ffmpeg", "-v", "error", "-i", source, "-frames:v", "1", "-vf",
              "scale=320:180,pad=320:192:0:0", "-pix_fmt", "nv12", "-f", "rawvideo", "-"]).stdout
    base = [binary, model, str(w), str(h), "2", "352", "224", "--margin", margin]
    if os.environ.get("SR_INPUT_INT16") == "1":
        base.append("--input-int16")

    def infer(data, *args):
        return run(base + list(args), input=data).stdout

    for label, data, opts in [("GBR", gbr, []), ("NV12 padded", nv, ["--in-nv12", "--in-h", "192"])]:
        raw = infer(data, *opts, "--nv12", "--preserve-bottom", "0.2")
        gpu = infer(data, *opts, "--nv12", "--preserve-bottom", "0.2", "--gpu")
        fp = infer(data, *opts, "--nv12", "--preserve-bottom", "0.2", "--float-out")
        assert len(raw) == w * h * 4 * 3 // 2
        assert raw == gpu == fp, label + ": CPU/GPU/float output mismatch"
        for gpu_in, name in [(False, "cpu"), (True, "gpu")]:
            path = tmp / (name + ".tiles")
            infer(data, *opts, *(["--gpu-in"] if gpu_in else []), "--dump-tiles", str(path))
        assert (tmp / "cpu.tiles").read_bytes() == (tmp / "gpu.tiles").read_bytes()
        # The protected band must equal direct bilinear interpolation of source
        # luminance, independently of neural model output. Match integer rounding.
        if label.startswith("NV12"):
            ysrc = np.frombuffer(data, np.uint8, count=w * 192).reshape(192, w)[:h].astype(np.int32)
        else:
            g, b, r = np.frombuffer(data, np.uint8).reshape(3, h, w).astype(np.int32)
            ysrc = ((16829 * r + 33039 * g + 6416 * b) >> 16) + 16
        ysrc = np.clip(ysrc, 16, 235)
        out = np.frombuffer(raw, np.uint8, count=w * h * 4).reshape(h * 2, w * 2)
        xq = np.clip((np.arange(w * 2) * 2 + 1) * 64 - 128, 0, (w - 1) * 256)
        for y in range(h * 2 - 40, h * 2):
            yq = min(max((y * 2 + 1) * 64 - 128, 0), (h - 1) * 256)
            x0, fx = xq >> 8, xq & 255
            x1 = np.minimum(x0 + 1, w - 1)
            y0, fy = yq >> 8, yq & 255
            y1 = min(y0 + 1, h - 1)
            a = ysrc[y0, x0] * (256 - fx) + ysrc[y0, x1] * fx
            b = ysrc[y1, x0] * (256 - fx) + ysrc[y1, x1] * fx
            reference = (a * (256 - fy) + b * fy + 32768) >> 16
            assert np.array_equal(out[y], reference), label + ": source strokes changed"
        print("PASS", label, "CPU/GPU/float, tile channels, original bottom luminance")

    for opts in [["--unknown-option"], ["--preserve-bottom", "nan", "--nv12"]]:
        result = subprocess.run(base + opts, input=b"", capture_output=True)
        assert result.returncode != 0, "invalid option silently accepted"
    wrong_type = [x for x in base if x != "--input-int16"] if "--input-int16" in base else base + ["--input-int16"]
    result = subprocess.run(wrong_type, input=gbr, capture_output=True)
    assert result.returncode != 0 and "输入格式" in result.stderr.decode(), "wrong input dtype silently accepted"
    mismatch = base.copy()
    mismatch[4] = "4"
    result = subprocess.run(mismatch, input=gbr, capture_output=True)
    assert result.returncode != 0 and "NBG" in result.stderr.decode()
    print("PASS unknown options, NaN protection, input dtype and mismatched NBG scale are rejected")
