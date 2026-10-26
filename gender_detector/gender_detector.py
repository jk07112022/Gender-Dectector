"""
Live Webcam Human Detection + Gender Classification (DeepFace edition)
=======================================================================

Why this version: DeepFace downloads and caches its own verified model
weights automatically on first run (to ~/.deepface/weights/) -- no manual
hunting for .caffemodel/.prototxt files, no mismatched architecture pairs,
no Git LFS pointer issues.

------------------------------------------------------------------
SETUP
------------------------------------------------------------------
1) Install dependencies:
     pip install deepface opencv-python

   (DeepFace will pull in tensorflow automatically if it's not already
   installed. First install may take a few minutes.)

2) Run:
     python gender_detector.py
     python gender_detector.py --camera 1        # different camera index
     python gender_detector.py --video file.mp4  # analyze a video file instead

   On the very first run, DeepFace will silently download the face
   detector + gender/age/emotion model weights (a few hundred MB total).
   This only happens once; later runs start instantly.

Press 'q' to quit the live window.
------------------------------------------------------------------
"""

import sys
import time
import argparse

import cv2

try:
    from deepface import DeepFace
except ImportError:
    print("DeepFace is not installed. Run: pip install deepface opencv-python")
    sys.exit(1)

# Analyze every Nth frame -- gender/age analysis is heavier than plain face
# detection, so we don't need to run it on every single frame to feel "live".
ANALYZE_EVERY_N_FRAMES = 5


def draw_label(frame, box, text, color):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def analyze_frame(frame):
    """
    Runs DeepFace face-detection + gender analysis on a frame.
    Returns a list of dicts: {box: (x1,y1,x2,y2), gender: str, confidence: float}
    """
    results = []
    try:
        analyses = DeepFace.analyze(
            img_path=frame,
            actions=["gender"],
            detector_backend="opencv",   # fast, good default; try "retinaface" for higher accuracy (slower)
            enforce_detection=False,     # don't throw if no face found
            silent=True,
        )
    except Exception as e:
        print(f"Analysis error: {e}")
        return results

    # DeepFace.analyze returns a list of dicts (one per detected face)
    if isinstance(analyses, dict):
        analyses = [analyses]

    for face in analyses:
        region = face.get("region", {})
        x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
        if w <= 0 or h <= 0:
            continue

        gender_scores = face.get("gender", {})  # e.g. {"Man": 12.3, "Woman": 87.7}
        if not gender_scores:
            continue
        dominant_gender = face.get("dominant_gender", max(gender_scores, key=gender_scores.get))
        confidence = gender_scores.get(dominant_gender, 0.0)

        results.append({
            "box": (x, y, x + w, y + h),
            "gender": dominant_gender,
            "confidence": confidence,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Live human detection + gender classification (DeepFace)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to a video file instead of a live camera")
    args = parser.parse_args()

    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open video source: {source}")
        sys.exit(1)

    print("Warming up DeepFace (first run downloads model weights, may take a bit)...")

    prev_time = time.time()
    fps = 0.0
    frame_count = 0
    last_results = []

    print("Running. Press 'q' in the video window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % ANALYZE_EVERY_N_FRAMES == 0:
            last_results = analyze_frame(frame)

        for face in last_results:
            label = face["gender"]
            conf = face["confidence"]
            color = (255, 140, 0) if label == "Man" else (180, 0, 200)
            text = f"{label} ({conf:.0f}%)"
            draw_label(frame, face["box"], text, color)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces: {len(last_results)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Human Detection + Gender Classification (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
