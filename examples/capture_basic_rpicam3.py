"""
capture_basic_rpicam3.py

Basic camera test for RPi Camera Module 3 (Picamera2) using the
producer / consumer pattern.

    [camera] --producer thread--> [queue (latest frame only)] --consumer--> [process + display]

- Producer (thread)   : grabs frames from the camera as fast as the sensor delivers them.
- Consumer (main)     : takes the most recent frame, runs a blocking process(), displays it.
                        OpenCV GUI calls (imshow / waitKey) are kept in the main thread
                        because most HighGUI backends (Qt / GTK) are not thread-safe.

Because the queue holds only one frame, a slow process() never builds up latency:
old frames are dropped and the consumer always works on the freshest image.
Replace process() with your own pre-processing + Hailo inference later.

Press 'q' (or Ctrl+C) to quit.
"""
import argparse
import queue
import threading
import time

import cv2
from picamera2 import Picamera2
from libcamera import controls


def process(frame):
    # some blocking process (simulates pre-processing + model inference)
    time.sleep(0.2)
    return frame


def put_latest(q, item):
    """Put item into a size-1 queue, dropping the old item if present.
    Returns True if an old frame was dropped."""
    dropped = False
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
            dropped = True
        except queue.Empty:
            pass
        q.put_nowait(item)
    return dropped


def task_inputframe(picam2, q, stop_event, stats):
    """Producer: read frames from the camera and push them to the queue."""
    frame_id = 0
    t_prev = time.perf_counter()
    try:
        while not stop_event.is_set():
            # capture_array() blocks until the next frame and returns a copy,
            # so it is safe to hand over to another thread.
            frame = picam2.capture_array("main")
            frame_id += 1

            if put_latest(q, (frame_id, frame)):
                stats["dropped"] += 1

            t_now = time.perf_counter()
            stats["capture_fps"] = 0.9 * stats["capture_fps"] + 0.1 / max(t_now - t_prev, 1e-6)
            t_prev = t_now
    except Exception as e:
        if not stop_event.is_set():   # errors while closing the camera are expected
            print("capture error:", e)
    finally:
        stop_event.set()


def draw_info(frame, frame_id, stats, process_fps):
    text = [
        f"frame #{frame_id}",
        f"capture fps: {stats['capture_fps']:.1f}",
        f"process fps: {process_fps:.1f}",
        f"dropped    : {stats['dropped']}",
    ]
    for i, t in enumerate(text):
        cv2.putText(frame, t, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="RPi Camera Module 3 producer/consumer test")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    picam2 = Picamera2()
    # Video configuration (not still configuration) is meant for continuous streaming:
    # several buffers, fixed frame rate, fast capture.
    # Note: Picamera2 "RGB888" is stored in B,G,R byte order -> already what OpenCV expects.
    config = picam2.create_video_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"},
        controls={"FrameRate": args.fps},
    )
    picam2.configure(config)

    # Camera Module 3 has autofocus: enable continuous AF if the sensor supports it
    if "AfMode" in picam2.camera_controls:
        picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

    q = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    stats = {"capture_fps": 0.0, "dropped": 0}

    picam2.start()
    t1 = threading.Thread(target=task_inputframe, args=(picam2, q, stop_event, stats), daemon=True)
    t1.start()

    print("start capture")
    process_fps = 0.0
    t_prev = time.perf_counter()
    try:
        while not stop_event.is_set():
            # Consumer: wait for a frame (timeout so we can still react to stop_event)
            try:
                frame_id, frame = q.get(timeout=0.5)
            except queue.Empty:
                continue

            # some blocking process
            out = process(frame)

            t_now = time.perf_counter()
            process_fps = 0.9 * process_fps + 0.1 / max(t_now - t_prev, 1e-6)
            t_prev = t_now

            # output / display
            draw_info(out, frame_id, stats, process_fps)
            cv2.imshow("output", out)

            # check keyboard for exit program
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        t1.join(timeout=2.0)
        picam2.stop()
        picam2.close()
        cv2.destroyAllWindows()
        print("capture is over")


if __name__ == "__main__":
    main()
