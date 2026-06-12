from ultralytics import YOLO
import cv2
import pyttsx3
import time
import collections

# =====================================
# LOAD TRAINED MODEL
# =====================================

model = YOLO("best.pt")

# Verify class mapping on startup
print("\n[INFO] Model Class Mapping:")
for idx, name in model.names.items():
    print(f"  {idx}: {name}")
print()

# =====================================
# INITIALIZE VOICE ENGINE
# =====================================

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# Optional: Use a clearer voice if available
voices = engine.getProperty('voices')
if voices:
    engine.setProperty('voice', voices[0].id)

# =====================================
# CONFIGURATION
# =====================================

CONF_THRESHOLD   = 0.75   # Minimum confidence to accept a detection
IOU_THRESHOLD    = 0.45   # NMS IoU threshold (suppress overlapping boxes)
SPEECH_DELAY     = 3.0    # Seconds before repeating the same announcement
STABILITY_FRAMES = 5      # Frames a class must appear before announcing

# =====================================
# OPEN WEBCAM
# =====================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Cannot open webcam.")
    exit()

# Optionally improve frame quality
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_AUTOFOCUS,    1)

# =====================================
# STATE TRACKING
# =====================================

last_spoken      = ""
last_spoken_time = 0.0

# Stability buffer: track recent detections to avoid flickering announcements
# Keeps the last N detected class names; announces only when one class dominates
detection_buffer = collections.deque(maxlen=STABILITY_FRAMES)

# =====================================
# HELPER FUNCTIONS
# =====================================

def get_stable_detection(buffer):
    """
    Returns a class name only when it appears in ALL recent frames,
    preventing flicker-based false positives.
    """
    if len(buffer) < buffer.maxlen:
        return None  # Not enough frames yet

    counts = collections.Counter(buffer)
    top_class, top_count = counts.most_common(1)[0]

    if top_count == buffer.maxlen:
        return top_class

    return None


def speak(text):
    """Non-blocking TTS announcement."""
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[TTS ERROR] {e}")


def draw_overlay(frame, text, pos, font_scale=0.8,
                 color=(0, 255, 255), thickness=2):
    """Draws text with a dark shadow for better readability."""
    x, y = pos
    # Shadow
    cv2.putText(frame, text, (x + 2, y + 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), thickness + 1, cv2.LINE_AA)
    # Main text
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                color, thickness, cv2.LINE_AA)


# =====================================
# MAIN LOOP
# =====================================

print("[INFO] Starting detection. Press Q to quit.\n")

while True:

    success, frame = cap.read()

    if not success:
        print("[WARNING] Failed to read frame. Retrying...")
        time.sleep(0.1)
        continue

    # ------------------------------------------------------------------
    # Run YOLO inference with strict thresholds
    # ------------------------------------------------------------------
    results = model(
        frame,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False          # Suppress per-frame YOLO console spam
    )

    annotated_frame = results[0].plot()
    detections      = results[0].boxes

    # ------------------------------------------------------------------
    # Process detections
    # ------------------------------------------------------------------
    best_class      = None
    best_confidence = 0.0

    print(f"--- Frame | {len(detections)} detection(s) ---")

    for box in detections:

        confidence = float(box.conf[0])

        # Hard filter (redundant safety net on top of model conf param)
        if confidence < CONF_THRESHOLD:
            continue

        class_id   = int(box.cls[0])
        class_name = model.names[class_id]

        print(f"  Class: {class_name:>10} | Conf: {confidence:.4f}")

        # Track the highest-confidence detection this frame
        if confidence > best_confidence:
            best_confidence = confidence
            best_class      = class_name

    if best_class is None:
        print("  No confident detection this frame.")

    # ------------------------------------------------------------------
    # Stability check — only announce a rock-solid detection
    # ------------------------------------------------------------------
    detection_buffer.append(best_class if best_class else "none")

    stable_class = get_stable_detection(detection_buffer)

    current_time = time.time()

    if (
        stable_class
        and stable_class != "none"
        and (
            stable_class != last_spoken
            or current_time - last_spoken_time > SPEECH_DELAY
        )
    ):
        speech_text = f"{stable_class} rupees detected"
        print(f"\n[SPEECH] {speech_text}\n")

        speak(speech_text)

        last_spoken      = stable_class
        last_spoken_time = current_time

    # ------------------------------------------------------------------
    # HUD overlay
    # ------------------------------------------------------------------
    draw_overlay(annotated_frame,
                 "AI Currency Detection System",
                 (20, 40), font_scale=1.0, color=(0, 255, 255))

    draw_overlay(annotated_frame,
                 "Press Q to Exit",
                 (20, 80), font_scale=0.7, color=(255, 255, 255))

    # Show confidence of best detection
    if best_class:
        status = f"Detecting: {best_class}  ({best_confidence:.0%})"
        color  = (0, 255, 0) if best_confidence >= 0.85 else (0, 165, 255)
    else:
        status = "No currency detected"
        color  = (100, 100, 100)

    draw_overlay(annotated_frame, status,
                 (20, annotated_frame.shape[0] - 20),
                 font_scale=0.75, color=color)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    cv2.imshow("Currency Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n[INFO] Exiting...")
        break

# =====================================
# RELEASE RESOURCES
# =====================================

cap.release()
cv2.destroyAllWindows()
print("[INFO] Resources released. Goodbye.")