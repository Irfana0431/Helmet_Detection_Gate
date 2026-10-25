import cv2
import serial
import time
from ultralytics import YOLO

# ---------------- CONFIGURATION ----------------

HELMET_MODEL_PATH = "best.pt"
PERSON_MODEL_PATH = "yolo26n.pt"

ARDUINO_PORT = "COM7"
BAUD_RATE = 9600

HELMET_CONFIDENCE = 0.35
PERSON_CONFIDENCE = 0.25
MOTORCYCLE_CONFIDENCE = 0.25

OPEN_DURATION = 5
REQUIRED_FRAMES = 3
CAMERA_INDEX = 0

# ------------------------------------------------


def normalize_label(label):
    return (
        str(label).lower()
        .strip()
        .replace("-", "_")
        .replace(" ", "_")
    )


def get_helmet_type(label):
    """Recognize helmet and no-helmet class-name variations."""

    label = normalize_label(label)

    no_helmet_names = {
        "no_helmet",
        "nohelmet",
        "without_helmet",
        "withouthelmet",
        "not_wearing_helmet",
        "no_helmet_on",
    }

    helmet_names = {
        "helmet",
        "with_helmet",
        "wearing_helmet",
    }

    if label in no_helmet_names:
        return "no_helmet"

    if label in helmet_names:
        return "helmet"

    return None


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def intersection_area(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    return max(0, x2 - x1) * max(0, y2 - y1)


def is_person_on_motorcycle(person_box, bike_box):
    """
    Associate a person with a motorcycle using geometry.
    This is an approximation and should be tested with
    the actual camera angle and installation.
    """

    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = bike_box

    bike_width = max(1, bx2 - bx1)
    bike_height = max(1, by2 - by1)

    # Expand motorcycle region upwards to include riders.
    expanded_left = bx1 - 0.45 * bike_width
    expanded_right = bx2 + 0.45 * bike_width
    expanded_top = by1 - 2.0 * bike_height
    expanded_bottom = by2 + 0.25 * bike_height

    # Use the person's lower-body center.
    foot_x = (px1 + px2) / 2
    foot_y = py2

    inside_expanded_region = (
        expanded_left <= foot_x <= expanded_right
        and expanded_top <= foot_y <= expanded_bottom
    )

    overlap = intersection_area(person_box, bike_box)
    person_area = max(1, box_area(person_box))
    overlap_ratio = overlap / person_area

    return inside_expanded_region or overlap_ratio >= 0.15


def detect_helmet_objects(frame, helmet_model, class_names):
    detections = []

    results = helmet_model.predict(
        source=frame,
        conf=HELMET_CONFIDENCE,
        verbose=False
    )

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            label = class_names.get(class_id, "")
            helmet_type = get_helmet_type(label)

            if helmet_type is None:
                continue

            coordinates = tuple(
                map(int, box.xyxy[0].tolist())
            )

            detections.append({
                "box": coordinates,
                "type": helmet_type,
                "confidence": confidence,
            })

    return detections


def detect_people_and_bikes(frame, general_model):
    """
    COCO pretrained YOLO model:
    class 0 = person
    class 3 = motorcycle
    """

    results = general_model.predict(
        source=frame,
        conf=PERSON_CONFIDENCE,
        classes=[0, 3],
        verbose=False
    )

    people = []
    motorcycles = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            coordinates = tuple(
                map(int, box.xyxy[0].tolist())
            )

            if class_id == 0:
                people.append({
                    "box": coordinates,
                    "confidence": confidence,
                })

            elif class_id == 3:
                if confidence >= MOTORCYCLE_CONFIDENCE:
                    motorcycles.append({
                        "box": coordinates,
                        "confidence": confidence,
                    })

    return people, motorcycles


def associate_riders_with_helmets(riders, helmet_detections):
    """
    Match helmet/no-helmet detections to individual riders.
    Only detections around the upper head region are considered.
    """

    possible_matches = []

    for rider_index, rider in enumerate(riders):
        x1, y1, x2, y2 = rider["box"]

        width = max(1, x2 - x1)
        height = max(1, y2 - y1)

        head_center_x = (x1 + x2) / 2
        head_center_y = y1 + 0.18 * height

        for detection_index, detection in enumerate(
            helmet_detections
        ):
            hx1, hy1, hx2, hy2 = detection["box"]

            detection_center_x = (hx1 + hx2) / 2
            detection_center_y = (hy1 + hy2) / 2

            # Detection must be around the upper part
            # of the person's bounding box.
            if not (
                x1 - 0.20 * width
                <= detection_center_x
                <= x2 + 0.20 * width
            ):
                continue

            if not (
                y1 - 0.15 * height
                <= detection_center_y
                <= y1 + 0.45 * height
            ):
                continue

            dx = abs(
                detection_center_x - head_center_x
            ) / width

            dy = abs(
                detection_center_y - head_center_y
            ) / height

            distance = dx + dy

            # Reject detections too far from the head region.
            if distance > 0.65:
                continue

            score = distance - (
                0.10 * detection["confidence"]
            )

            possible_matches.append((
                score,
                rider_index,
                detection_index
            ))

    # Each rider and each detection can be assigned once.
    possible_matches.sort(key=lambda item: item[0])

    assignments = {}
    used_riders = set()
    used_detections = set()

    for _, rider_index, detection_index in possible_matches:
        if rider_index in used_riders:
            continue

        if detection_index in used_detections:
            continue

        assignments[rider_index] = helmet_detections[
            detection_index
        ]

        used_riders.add(rider_index)
        used_detections.add(detection_index)

    return assignments

def filter_riders(riders, bike_box):
    bike_area = max(1, box_area(bike_box))

    # Remove boxes much larger than the bike
    riders = [
        r for r in riders
        if box_area(r["box"]) <= 2.5 * bike_area
    ]

    # Highest confidence first
    riders.sort(key=lambda r: r["confidence"], reverse=True)

    kept = []
    for r in riders:
        duplicate = False
        rx1, ry1, rx2, ry2 = r["box"]
        r_cx = (rx1 + rx2) / 2

        for k in kept:
            kx1, ky1, kx2, ky2 = k["box"]
            k_cx = (kx1 + kx2) / 2

            inter = intersection_area(r["box"], k["box"])
            smaller_area = max(1, min(box_area(r["box"]), box_area(k["box"])))
            smaller_width = max(1, min(rx2 - rx1, kx2 - kx1))

            mostly_inside = inter / smaller_area > 0.7
            same_center = abs(r_cx - k_cx) < 0.25 * smaller_width

            if mostly_inside and same_center:
                duplicate = True
                break

        if not duplicate:
            kept.append(r)

    return kept

def main():
    helmet_model = YOLO(HELMET_MODEL_PATH)
    general_model = YOLO(PERSON_MODEL_PATH)

    class_names = {
        int(class_id): str(name)
        for class_id, name in helmet_model.names.items()
    }

    detected_types = {
        get_helmet_type(name)
        for name in class_names.values()
    }

    if not {"helmet", "no_helmet"}.issubset(
        detected_types
    ):
        raise RuntimeError(
            "best.pt must contain helmet and no_helmet "
            f"classes. Found: {class_names}"
        )

    arduino = None
    camera = None

    helmet_frames = 0
    gate_open = False
    gate_open_time = 0

    try:
        # Connect to Arduino.
        arduino = serial.Serial(
            ARDUINO_PORT,
            BAUD_RATE,
            timeout=1
        )

        time.sleep(2)

        # Start webcam.
        camera = cv2.VideoCapture(CAMERA_INDEX)

        if not camera.isOpened():
            raise RuntimeError("Could not open webcam")

        print("Helmet detection system started.")
        print("Press Q to exit.")

        while True:
            success, frame = camera.read()
            if not success:
                helmet_frames = 0
                print("Camera frame unavailable.")
                break

            # 1. Detect people and motorcycles.
            people, motorcycles = detect_people_and_bikes(
                frame, general_model
            )
            # 2. Detect helmet/no-helmet objects.
            helmet_detections = detect_helmet_objects(
                frame,
                helmet_model,
                class_names
            )

            # Display motorcycle boxes.
            for bike in motorcycles:
                x1, y1, x2, y2 = bike["box"]

                cv2.rectangle(
                    frame, (x1, y1), (x2, y2),
                    (255, 180, 0), 2
                )

                cv2.putText(
                    frame, "Motorcycle",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 180, 0), 2
                )

            # Select the largest detected motorcycle
            # as the vehicle at the gate.
            selected_bike = None

            if motorcycles:
                selected_bike = max(
                    motorcycles,
                    key=lambda bike: box_area(bike["box"])
                )

            riders = []

            if selected_bike is not None:
                riders = [
                    person for person in people
                    if is_person_on_motorcycle(
                        person["box"],
                        selected_bike["box"]
                    )
                ]
                riders = filter_riders(riders, selected_bike["box"])

            # 3. Associate helmet detections with riders.
            assignments = associate_riders_with_helmets(
                riders, helmet_detections
            )

            all_riders_safe = (
                selected_bike is not None
                and len(riders) in (1, 2)
                and len(assignments) == len(riders)
                and all(
                    assignments[i]["type"] == "helmet"
                    for i in range(len(riders))
                )
            )

            # 4. Draw individual rider status.
            for index, rider in enumerate(riders):
                x1, y1, x2, y2 = rider["box"]

                assigned = assignments.get(index)

                if assigned is None:
                    label = "HELMET UNCERTAIN"
                    color = (0, 165, 255)

                elif assigned["type"] == "helmet":
                    label = "HELMET"
                    color = (0, 255, 0)

                else:
                    label = "NO HELMET"
                    color = (0, 0, 255)

                cv2.rectangle(
                    frame, (x1, y1), (x2, y2),
                    color, 2
                )

                cv2.putText(
                    frame,
                    f"Rider {index + 1}: {label}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2
                )

            # Draw helmet detector boxes for debugging.
            for detection in helmet_detections:
                x1, y1, x2, y2 = detection["box"]

                if detection["type"] == "helmet":
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                cv2.rectangle(
                    frame, (x1, y1), (x2, y2),
                    color, 1
                )

            # 5. Require consecutive safe frames.
            if all_riders_safe:
                helmet_frames += 1
            else:
                helmet_frames = 0

            # 6. Open gate only after all riders pass.
            if helmet_frames >= REQUIRED_FRAMES and not gate_open:
                arduino.write(b"1")
                arduino.flush()

                gate_open = True
                gate_open_time = time.time()
                helmet_frames = 0

                print(
                    f"All {len(riders)} rider(s) verified. "
                    "Gate opening."
                )

            # 7. Close gate after the configured duration.
            if (
                gate_open
                and time.time() - gate_open_time >= OPEN_DURATION
            ):
                arduino.write(b"0")
                arduino.flush()

                gate_open = False
                print("Gate closing.")

            # 8. Display verification status.
            if gate_open:
                status = "GATE OPEN"
                status_color = (0, 255, 0)

            elif selected_bike is None:
                status = "NO MOTORCYCLE"
                status_color = (0, 165, 255)

            elif len(riders) not in (1, 2):
                status = "INVALID RIDER COUNT"
                status_color = (0, 0, 255)

            elif all_riders_safe:
                status = (
                    f"VERIFYING {helmet_frames}/"
                    f"{REQUIRED_FRAMES}"
                )
                status_color = (0, 255, 255)

            else:
                status = "GATE CLOSED - CHECK HELMETS"
                status_color = (0, 0, 255)

            cv2.putText(
                frame, status, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                status_color, 2
            )

            cv2.imshow(
                "AI Helmet Detection Smart Gate", frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Always command the gate to close on exit.
        if arduino is not None and arduino.is_open:
            try:
                arduino.write(b"0")
                arduino.flush()
            except serial.SerialException:
                pass

            arduino.close()

        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()