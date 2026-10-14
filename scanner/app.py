"""Command-line entry point.

    python -m scanner.app                 live preview from the connected camera
    python -m scanner.app --simulate      live preview from the simulated camera
    python -m scanner.app list            list the cameras on the network
    python -m scanner.app capture         save one triggered frame and exit

The move-settle-capture scan sequence will be added here once the robot
interface exists.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .camera import (Camera, CameraError, HeliosCamera, SimulatedCamera, capture_folder,
                     capture_name, save_capture)

DEFAULT_CAPTURE_DIR = Path("data")


def build_parser() -> argparse.ArgumentParser:
    source = argparse.ArgumentParser(add_help=False)
    source.add_argument("--simulate", action="store_true",
                        help="use synthetic data instead of a camera")
    source.add_argument("--serial", help="serial number of the camera to open")
    source.add_argument("--out", type=Path, default=DEFAULT_CAPTURE_DIR,
                        help="folder for saved captures (default: data)")

    parser = argparse.ArgumentParser(prog="python -m scanner.app", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    preview = commands.add_parser("preview", parents=[source], help="live preview window")
    preview.add_argument("--duration", type=float, metavar="SECONDS",
                         help="close the window automatically after this long")
    preview.add_argument("--screenshot", type=Path, metavar="PNG",
                         help="with --duration, save the window to this image before closing")

    capture = commands.add_parser("capture", parents=[source], help="save one frame and exit")
    capture.add_argument("--free-run", action="store_true",
                         help="take a frame from a free-running stream instead of "
                              "sending a software trigger")
    capture.add_argument("--native-ply", action="store_true",
                         help="also have the Arena SDK write the frame to a PLY file")

    commands.add_parser("list", help="list the cameras on the network")
    return parser


def build_camera(args) -> Camera:
    return SimulatedCamera() if args.simulate else HeliosCamera(serial=args.serial)


def command_list(args) -> int:
    cameras = HeliosCamera.discover()
    if not cameras:
        print("No cameras found.")
        return 1
    for camera in cameras:
        print(f"{camera.model}  serial {camera.serial}  ip {camera.ip}  "
              f"firmware {camera.firmware}")
    return 0


def command_capture(args) -> int:
    with build_camera(args) as camera:
        name = capture_name()
        native_ply = None
        if args.native_ply and camera.supports_native_ply:
            native_ply = capture_folder(args.out, name) / f"{name}_arena.ply"
        if args.free_run:
            camera.start()
            # Drop one frame so the one kept was exposed after the stream started.
            frame = camera.grab(discard=1, native_ply=native_ply)
        else:
            camera.start(triggered=True)
            frame = camera.trigger(native_ply=native_ply)
        if "native_ply_error" in frame.extra:
            print(f"Arena PLY export failed: {frame.extra['native_ply_error']}")
            native_ply = None
        paths = save_capture(frame, args.out, name, camera_info=camera.info,
                             settings=camera.read_settings(), native_ply=native_ply)
    print(f"Frame {frame.frame_id}: {frame.valid_count} valid points "
          f"({100 * frame.valid_fraction:.1f}%)")
    for path in (paths.npz, paths.metadata, paths.ply, paths.native_ply):
        if path is not None:
            print(f"  {path}")
    return 0


def command_preview(args) -> int:
    from .ui import run_preview

    return run_preview(build_camera(args), args.out, screenshot=args.screenshot,
                       duration_s=args.duration)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # The live preview is the default command.
    if not argv or (argv[0].startswith("-") and argv[0] not in ("-h", "--help")):
        argv.insert(0, "preview")
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("OpenGL").setLevel(logging.WARNING)
    command = {"preview": command_preview, "capture": command_capture,
               "list": command_list}[args.command]
    try:
        return command(args)
    except CameraError as exc:
        print(f"Camera error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
