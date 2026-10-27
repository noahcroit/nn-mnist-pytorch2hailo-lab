# Guide: YOLOv5n ONNX → Hailo-8 HEF (Hailo Dataflow Compiler)

How `yolov5n.onnx` was converted to `yolov5n.hef` for the Hailo-8 on the Raspberry Pi 5,
with the problems found on the way and how they were solved.

Tested with: Hailo DFC **3.34.0**, Python 3.10 venv (`~/SoftwareDev/venv/hailo_dfc`), x86 PC with
**no GPU** (AMD Ryzen 7 260, 14 GB RAM), target `hailo8`.

---

## 0. Overview

```
yolov5n.onnx ──parse──> yolov5n.har ──optimize──> yolov5n_optimized.har ──compile──> yolov5n.hef
   (float)              (Hailo model)  + .alls      (8-bit quantized)                 (for the Pi)
                                       + calibration set
```

| Step | Script | Input | Output | Time (CPU only) |
|---|---|---|---|---|
| 1. Parse | `onnx2har.py` | `../models/yolov5n.onnx` | `output/yolov5n.har` | seconds |
| 2. Calibration set | `make_calib_yolo.py` | photos | `../calib_set_yolo.npy` | ~1 min |
| 3. Optimize | `optimize.py` + `yolov5n.alls` | HAR + calibration set | `output/yolov5n_optimized.har` | ~3 min |
| 4. Compile | `compile.py` | optimized HAR | `output/yolov5n.hef` | ~1-2 min |

All four scripts read the same config file, `yolov5n_config.json`. Run every command from the
`compilation/` folder, because the paths in the config are relative to it.

```bash
source ~/SoftwareDev/venv/hailo_dfc/bin/activate
cd ~/Claude_Sandbox/Hailo-AI-Kit/compilation
```

---

## 1. The ONNX model

Export it with Ultralytics yolov5 using a **static** 640×640 input in **float32**:

```bash
python export.py --weights yolov5n.pt --include onnx --imgsz 640 --opset 17
cp yolov5n.onnx ~/Claude_Sandbox/Hailo-AI-Kit/models/
```

- Do **not** use `--dynamic`. A dynamic input (`[batch, 3, height, width]`) makes the parser fail with
  `Unsupported dynamic shape([0, 3, 0, 0]) found on input node images`.
  This is what happened with the earlier `yolov5s.onnx`.
- Do **not** use `--half`. The DFC needs float32.

To check the model, open it in Netron, or run in the DFC venv:

```python
import onnx
m = onnx.load("../models/yolov5n.onnx")
print(m.graph.input[0])     # images [1, 3, 640, 640]
print(m.graph.output[0])    # output0 [1, 25200, 85]
```

---

## 2. Parse (ONNX → HAR)

```bash
python onnx2har.py -c yolov5n_config.json
```

`yolov5n_config.json`:

```json
{
	"hailo_arch":"hailo8",
	"onnx_path":"../models/yolov5n.onnx",
	"model_name":"yolov5n",
	"name_start":"images",
	"name_end":["/model.24/m.0/Conv", "/model.24/m.1/Conv", "/model.24/m.2/Conv"],
	"size_input":"1,3,640,640",
	"output_path":"output/",
	"har_name":"yolov5n.har",
	"quantized_har_name":"yolov5n_optimized.har",
	"model_script":"yolov5n.alls",
	"calib_set":"../calib_set_yolo.npy"
}
```

### The important point: the end nodes

**Do not use `output0` as the end node.** After the last 3 Conv layers, the YOLOv5 Detect head
(`/model.24`) decodes the boxes with Reshape / Add / Mul operations on 5D tensors. The Hailo cannot
run these, and parsing fails:

```
UnsupportedShuffleLayerError in op /model.24/Reshape_1: Failed to determine type of layer to create
UnsupportedModelError in op /model.24/Add: ... constant value shape (1, 3, 80, 80, 2) must be broadcastable ...
```

Cut the network at the **3 last Conv layers of the Detect head**, one for each scale:

| End node | Output (NHWC) | Grid / stride |
|---|---|---|
| `/model.24/m.0/Conv` | 80 × 80 × 255 | stride 8, small objects |
| `/model.24/m.1/Conv` | 40 × 40 × 255 | stride 16 |
| `/model.24/m.2/Conv` | 20 × 20 × 255 | stride 32, large objects |

255 = 3 anchors × (4 box + 1 objectness + 80 classes). The part that was cut off (the sigmoid, box
decoding and NMS) is added back in the optimize step by the model script (`.alls`).

YOLOv5n, s, m and l all use the same names. For a different model, look up the last Conv layers in Netron.

A successful parse prints:

```
[info] NMS structure of yolov5 (or equivalent architecture) was detected.
[info] End nodes mapped from original model: '/model.24/m.0/Conv', '/model.24/m.1/Conv', '/model.24/m.2/Conv'.
[info] Saved HAR to: .../output/yolov5n.har
```

> CLI alternative: `hailo parser onnx ../models/yolov5n.onnx --hw-arch hailo8 --har-path output/yolov5n.har -y`
> (`-y` accepts the end nodes the parser recommends, but these are not exactly the Conv nodes above).
> The script with the config is the controlled way.

---

## 3. Calibration set

The optimizer runs real images through the model to measure the range of values in every layer, and
uses these ranges to convert float32 to 8-bit.

```bash
# COCO val2017 (5000 images, ~800 MB): http://images.cocodataset.org/zips/val2017.zip
# unzipped into ../img_yolo/val2017
python make_calib_yolo.py -i ../img_yolo/val2017 -o ../calib_set_yolo.npy
# -> saved ../calib_set_yolo.npy: shape (1024, 640, 640, 3), 1258 MB
```

Rules for the calibration images:

- **The same preprocessing as on the Pi.** `make_calib_yolo.py` uses the same letterbox as
  `frame_preprocess()` in `inference_yolov5_*.py`: resize with the aspect ratio kept, gray (114)
  padding, RGB order.
- **Values 0..255, not normalized.** The normalization (÷255) is in the `.alls`, so the chip does it.
  Normalizing the calibration set as well would divide twice.
- **Images like the real use.** COCO is fine for the COCO-trained model. Photos from the Pi camera
  of the real scene are even better.
- **uint8 storage.** The DFC accepts a uint8 `.npy` (tested). 1024 images take 1.2 GB as uint8
  instead of 5 GB as float32. This matters on a nearly full disk.
- `-n` sets the maximum number of images (default 1024). Without a GPU the DFC uses only 64 of them
  anyway (see step 4), so 1024 images are not needed on a CPU-only PC. A few hundred is enough.

---

## 4. Optimize (quantization)

```bash
python optimize.py -c yolov5n_config.json
```

`optimize.py` loads the model script named by `"model_script"` in the config, then runs
`runner.optimize(calib_set)`.

### The model script `yolov5n.alls`

```
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
change_output_activation(sigmoid)
nms_postprocess(meta_arch=yolov5, engine=cpu, nms_scores_th=0.2, nms_iou_th=0.6)
model_optimization_flavor(optimization_level=1)
```

| Line | What it does | Why |
|---|---|---|
| `normalization(mean=0, std=255)` | Adds an input layer `(x - 0) / 255` on the chip | The Pi sends the raw uint8 image (4× less PCIe data than float32), and the chip does the ÷255 |
| `change_output_activation(sigmoid)` | Adds the sigmoid back to the 3 outputs | It was in the cut-off part of the Detect head |
| `nms_postprocess(meta_arch=yolov5, engine=cpu, ...)` | Adds box decoding (anchors, grid) and NMS to the HEF | The Pi gets a list of final boxes per class instead of raw tensors |
| `model_optimization_flavor(optimization_level=1)` | Forces optimization level 1 | Without it, the DFC drops to level 0 on a PC without a GPU |

### `engine=cpu`, not `nn_core`

With `engine=nn_core`, the DFC warns:

```
[warning] Currently `nn-core` performs only score threshold. ... If you want to apply IoU threshold please use `cpu` or `auto`
```

With `nn_core`, the overlapping boxes are **not removed**, so every object gets several boxes.
With `engine=cpu`, HailoRT on the Pi does the decoding and NMS in C++, right after the NPU and before
the result reaches Python. The output format is the same either way, and the Hailo Model Zoo YOLOv5
HEFs work the same way.

### NMS thresholds

- `nms_scores_th=0.2`: boxes below this score are dropped in the HEF. `inference_yolov5_hailo.py --conf`
  can only raise it (default 0.25), not lower it.
- `nms_iou_th=0.6`: the overlap threshold. It is fixed in the HEF; change it here and recompile.
- Without an NMS config JSON, the DFC uses the default YOLOv5 anchors and 80 classes, so this works
  for **COCO models only**. A custom-trained model needs its own NMS config file
  (`nms_postprocess("my_nms_config.json", meta_arch=yolov5, engine=cpu)`).

### Optimization level: what happens without a GPU

```
[warning] Reducing optimization level to 0 (the accuracy won't be optimized and compression won't be used)
          because there's no available GPU
[info] Using dataset with 64 entries for calibration
```

| Level | Algorithms | Needs | Time here |
|---|---|---|---|
| 0 | Statistics only (min/max ranges) | – | ~1 min |
| **1** | + Bias Correction | CPU is fine | **~3 min** |
| 2+ | + fine-tuning (QFT), AdaRound | GPU and 1024+ images | not practical on CPU |

Level 1 is forced with the last `.alls` line. The log then shows:

```
[info] Model Optimization Algorithm Bias Correction is done (completion time is 00:01:27)
[info] Model Optimization is done
```

### Measured accuracy (emulator, 50 COCO images not used for calibration)

The quantized model was compared with the float model (score ≥ 0.25, same class, IoU > 0.5):

| | Float objects also found by the 8-bit model | Average score change |
|---|---|---|
| Level 0 | 80% (214/269) | −0.028 |
| **Level 1** | **85% (228/269)** | **−0.010** |

YOLOv5n is very small, so it loses more from 8-bit quantization than larger models do. A GPU
(level 2) or a bigger model (YOLOv5s) gives better results.

---

## 5. Compile (optimized HAR → HEF)

```bash
python compile.py -c yolov5n_config.json
# -> output/yolov5n.hef (about 3.6 MB)
```

A successful compile prints:

```
[info] output_from_conv47_to_yolov5_nms_postprocess: Pass
...
[info] Successful Mapping (allocation time: 1m 4s)
[info] Successful Compilation (compilation time: 2s)
```

The resource table shows how much of the 8 clusters the model uses. YOLOv5n fits in a single context.

---

## 6. Check before deploying (optional)

```bash
# what the HAR contains (state, files)
hailo har info output/yolov5n_optimized.har

# quantization noise per layer (8-bit emulator vs float); output SNR above ~10 dB is usually OK
hailo analyze-noise output/yolov5n_optimized.har --data-path ../calib_set_yolo.npy --data-count 64

# expected FPS / latency / resources on Hailo-8 (HTML report)
hailo profiler output/yolov5n_optimized.har --hef output/yolov5n.hef
```

To see the detections themselves before deploying, run the emulator from Python:
`ClientRunner(har=...)`, then `runner.infer_context(InferenceContext.SDK_QUANTIZED)` and `runner.infer(ctx, images)`.
Use `SDK_FP_OPTIMIZED` for the float model with the `.alls` applied. If FP_OPTIMIZED is good but
QUANTIZED is bad, the quantization or calibration is the problem.

---

## 7. Deploy on the Raspberry Pi 5

```bash
# from the PC
scp output/yolov5n.hef ../models/yolov5n.onnx pi@<pi-address>:~/Hailo-AI-Kit/models/

# on the Pi
hailortcli fw-control identify      # Device Architecture: HAILO8
hailortcli --version                # same Hailo software release as DFC 3.34.0
python3 inference_yolov5_hailo.py -e models/yolov5n.hef --mode camera
python3 inference_yolov5_onnx.py  -e models/yolov5n.onnx --mode camera
```

What the HEF expects and returns (this matches `inference_yolov5_hailo.py`):

- **Input:** UINT8, NHWC 640×640×3, RGB, letterboxed (normalization is inside the HEF).
- **Output:** FLOAT32, one NMS output with 80 class lists of `[y0, x0, y1, x1, score]`,
  normalized 0..1 to the 640×640 input.

---

## 8. Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `Unsupported dynamic shape([0, 3, 0, 0])` | ONNX exported with `--dynamic` | Re-export with a static 640×640 input |
| `UnsupportedShuffleLayerError in op /model.24/Reshape_1` | End node `output0` includes the box decoding | End nodes = the 3 `/model.24/m.*/Conv` layers |
| Many boxes on the same object | `engine=nn_core` (score threshold only, no IoU) | `engine=cpu` in `nms_postprocess` |
| `Reducing optimization level to 0` | No GPU | Add `model_optimization_flavor(optimization_level=1)` |
| Few or no detections, very low scores | Calibration set normalized to 0..1 while the `.alls` also divides by 255 | Calibration set in 0..255 |
| Wrong colors / worse accuracy | Calibration in BGR, or no letterbox | RGB + letterbox, same as `frame_preprocess()` |
| Out of memory / disk while making the calibration set | float32 `.npy` (5 GB for 1024 images) | uint8 `.npy` (1.2 GB), or fewer images (`-n 256`) |
| `expected 80 classes in the NMS output` on the Pi | HEF without `nms_postprocess`, or a non-COCO model | Use the `.alls` above, COCO model |
| HEF does not load on the Pi | Compiled for another chip, or HailoRT ≠ DFC release | `--hw-arch hailo8` (Hailo-8L: `hailo8l`), matching versions |
| `Cannot use graphviz` warning | graphviz not installed | Harmless; `sudo apt install graphviz` for `hailo visualizer` |

---

## Files

| File | Purpose |
|---|---|
| `yolov5n_config.json` | Paths, model name, start / end nodes, input shape |
| `yolov5n.alls` | Model script: normalization, sigmoid, NMS, optimization level |
| `onnx2har.py` | Step 1: parse |
| `make_calib_yolo.py` | Step 2: calibration set (Pillow, already in the DFC venv) |
| `optimize.py` | Step 3: optimize (loads `model_script` from the config) |
| `compile.py` | Step 4: compile |
| `output/yolov5n.hef` | Result for the Pi |
