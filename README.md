# 3D Scanning Robot: camera software

Software for the ME 470 robotic 3D scanner: a LUCID Helios2+ Narrow
time-of-flight camera on a 6-axis arm, scanning an object from several
viewpoints and merging the views into one point cloud.

## Status

| Area | State |
|---|---|
| `scanner/camera` | Implemented: discovery, configuration, free-running and software-triggered acquisition, decoding to metric XYZ, saving to `.npz` + `.json` + `.ply` |
| `scanner/ui` | Implemented: live preview (intensity, depth map, 3D cloud) with capture |
| `scanner/robot` | Placeholder |
| `scanner/calibration` | Placeholder |
| `scanner/processing` | Placeholder |

**The camera code has not been run against a physical camera yet.** It was
developed against LUCID's `arena_api` source and examples and is tested with a
stand-in for that API and with a simulated camera. See
[First run with the real camera](#first-run-with-the-real-camera).

## Setup

Tested on Windows 11 with Python 3.12, Arena SDK 1.0.85.12 and `arena_api` 2.9.5.

1. Install the **Arena SDK** from LUCID's downloads page. It supplies the
   native libraries that `arena_api` loads, and ArenaView.
2. Download the **`arena_api` Python wheel** from the same page. It is licensed
   software, so it is not stored in this repository.
3. Create the environment and install everything:

   ```powershell
   py -3.12 -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt
   .venv\Scripts\python -m pip install path\to\arena_api-2.9.5-py3-none-any.whl
   ```

The camera needs a wired gigabit Ethernet port (or a USB adapter) and a PoE+
injector. Without a camera, everything except `scanner/camera/helios.py` still
runs, using `--simulate`.

## Running

```powershell
.venv\Scripts\python -m scanner.app                  # live preview, connected camera
.venv\Scripts\python -m scanner.app --simulate       # live preview, synthetic data
.venv\Scripts\python -m scanner.app list             # cameras found on the network
.venv\Scripts\python -m scanner.app capture          # save one triggered frame, no window
```

Options: `--serial <number>` picks a camera when several are connected, and
`--out <folder>` sets where captures go (default `data/`). `capture` also takes
`--free-run` (no trigger) and `--native-ply`.

Only one program can control the camera at a time: **disconnect ArenaView from
the camera before starting this application.**

### Live preview

* **Intensity**: the reflected illumination, shown like a greyscale photo. Use
  it to aim the camera.
* **Depth**: distance along the optical axis, near = bright. Black pixels have
  no valid depth. Hover over either image to read that pixel's X, Y, Z.
* **3D view**: the same frame as a point cloud in the camera's frame.
* **Status bar**: frame rate, frame ID, share of valid pixels, and the median
  distance at the image centre (the `+` mark).
* **Capture frame** saves the next frame to the output folder.
* **Camera settings** lists whichever tuning features the connected camera
  exposes (operating mode, exposure, accumulation, filters). Changing one
  briefly restarts the stream.

### What a capture contains

| File | Contents |
|---|---|
| `capture_<time>.npz` | Raw `(480, 640, 4)` uint16 payload, per-axis scale and offset, validity mask. The record of truth. |
| `capture_<time>.json` | Camera identity, settings, frame ID, timestamps, software versions, and a `robot_pose` slot for later. |
| `capture_<time>.ply` | Valid points in **metres**, camera frame, with an `intensity` property. |
| `capture_<time>_arena.ply` | Optional: the same frame written by LUCID's own library, in **millimetres**, for cross-checking. |

## Using the camera from code

```python
from scanner.camera import HeliosCamera, CameraSettings, save_capture, load_capture

settings = CameraSettings(image_accumulation=4, spatial_filter=True)
with HeliosCamera(settings=settings) as camera:
    camera.start(triggered=True)
    # ... move the robot, wait until it has settled ...
    frame = camera.trigger()                 # one frame per software trigger
    paths = save_capture(frame, "data", camera_info=camera.info,
                         settings=camera.read_settings())

frame = load_capture(paths.npz)              # works without a camera or the SDK
points = frame.points_m()                    # (N, 3) valid points, metres
```

`frame.xyz_m`, `frame.intensity` and `frame.valid` keep the camera's 480 x 640
pixel grid, which calibration will need. `SimulatedCamera` has the same
interface as `HeliosCamera`.

## Layout

```
scanner/
  camera/        arena_api wrapper: open, configure, trigger, decode, save
  robot/         serial comms to the Teensy, FK, pose readback   (to do)
  calibration/   hand-eye calibration with OpenCV                (to do)
  processing/    Open3D crop, filter, ICP, merge                 (to do)
  ui/            live preview; merged-cloud viewer later
  app.py         command-line entry point; scan orchestration later
data/            captures, not committed
tests/
```

All `arena_api` calls live in `scanner/camera/helios.py`, so the rest of the
code can be developed and tested on saved captures.

## Tests

```powershell
.venv\Scripts\python -m pytest
```

## First run with the real camera

Things to confirm on hardware, roughly in order:

1. Stream in ArenaView first to prove the network setup, then disconnect it.
2. `python -m scanner.app list` shows the camera.
3. `python -m scanner.app` opens it. If a feature name is rejected, the error
   lists the names the camera does accept.
4. The **Camera settings** panel shows the real option names for this model.
   Pick values there, then set them in `CameraSettings` for scans.
5. Point at a flat target at a measured distance and compare with the centre
   readout to confirm units and scale.
6. Capture with "Also save Arena-written PLY" ticked and compare the two PLY
   files (scale the Arena one by 0.001).
7. `python -m scanner.app capture` exercises the software trigger.

Assumptions taken from LUCID's examples that these steps will test: the
feature names used for configuration and triggering, raw Z = 65535 meaning "no
depth", and coordinates reported in millimetres.
