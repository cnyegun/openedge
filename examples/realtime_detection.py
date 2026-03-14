#!/usr/bin/env python3
"""
Test YOLOv8n INT8 model on laptop camera in real-time with bounding boxes.
Usage: python test_laptop_cam.py [model_path]
"""

import sys
import time
import numpy as np

try:
    import cv2
except ImportError:
    print("Error: opencv-python not installed.")
    print("Install with: pip install opencv-python")
    sys.exit(1)

try:
    import tensorflow as tf
except ImportError:
    print("Error: tensorflow not installed.")
    print("Install with: pip install tensorflow")
    sys.exit(1)

DEFAULT_MODEL = "/home/smooth/cless/artifacts/models/embedded/yolov8n_int8.tflite"

# COCO class names (80 classes)
COCO_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def letterbox(img, new_shape=(640, 640)):
    """Resize image with padding to maintain aspect ratio."""
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    return img, r, (dw, dh)


def xywh2xyxy(x):
    """Convert box format from [x, y, w, h] to [x1, y1, x2, y2]."""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y


def nms(dets, iou_threshold=0.45):
    """Apply NMS to detections with format [x1, y1, x2, y2, score, class]."""
    if len(dets) == 0:
        return np.zeros((0, 6))

    # Sort by score
    order = dets[:, 4].argsort()[::-1]
    dets = dets[order]

    keep = []
    while len(dets) > 0:
        keep.append(dets[0])
        if len(dets) == 1:
            break

        # Calculate IoU with remaining
        iou = box_iou(dets[0:1, :4], dets[1:, :4])
        dets = dets[1:][iou[0] < iou_threshold]

    return np.array(keep) if keep else np.zeros((0, 6))


def box_iou(box1, boxes):
    """Calculate IoU between box1 and boxes."""
    x1 = np.maximum(box1[..., 0:1], boxes[..., 0])
    y1 = np.maximum(box1[..., 1:2], boxes[..., 1])
    x2 = np.minimum(box1[..., 2:3], boxes[..., 2])
    y2 = np.minimum(box1[..., 3:4], boxes[..., 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])

    return inter / (area1 + area2 - inter + 1e-6)


def sigmoid(x):
    """Sigmoid activation (for ONNX models, TFLite already has sigmoid applied)."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def postprocess(output, orig_shape, new_shape, r, pad, debug=False):
    """Post-process YOLO output to get final boxes."""
    pred = output[0].T  # (8400, 84)

    boxes = pred[:, :4]
    class_scores = pred[:, 4:]  # raw logits

    # TFLite model already outputs sigmoid-activated class scores

    boxes = xywh2xyxy(boxes)

    # TFLite boxes are already normalized (0-1), scale to image size
    boxes[..., [0, 2]] *= orig_shape[1]  # x scale to image width
    boxes[..., [1, 3]] *= orig_shape[0]  # y scale to image height

    boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], 0, orig_shape[1])
    boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], 0, orig_shape[0])

    final_boxes = np.column_stack(
        [boxes, class_scores.max(axis=1), class_scores.argmax(axis=1)]
    )

    return final_boxes


def run_camera_test(model_path: str = DEFAULT_MODEL, conf_threshold: float = 0.25):
    print(f"Loading model: {model_path}")
    print(f"Confidence threshold: {conf_threshold}")

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print("\nOpening camera... (press 'q' to quit)")

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {width}x{height}")

    # Timing stats
    frame_times = []
    frame_count = 0
    last_fps_time = time.time()
    fps_display = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break

        orig_h, orig_w = frame.shape[:2]

        # Preprocess: letterbox resize, normalize to -1 to 1 (ImageNet style)
        img, ratio, pad = letterbox(frame, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = img * 2.0 - 1.0  # Convert 0-1 to -1 to 1
        img = np.expand_dims(img, axis=0)

        # Inference
        start_time = time.perf_counter()
        interpreter.set_tensor(input_index, img)
        interpreter.invoke()
        inference_time = (time.perf_counter() - start_time) * 1000

        # Get output
        output = interpreter.get_tensor(output_index)

        # Post-process
        debug = frame_count < 5  # Debug first few frames
        detections = postprocess(
            output, (orig_h, orig_w), (640, 640), ratio, pad, debug=debug
        )

        # Filter by confidence and apply NMS
        boxes = []
        if len(detections) > 0:
            # Get boxes and scores
            dets = detections[detections[:, 4] > conf_threshold]

            if len(dets) > 0:
                # Apply NMS
                nms_dets = nms(dets, iou_threshold=0.45)

                boxes = nms_dets[:, :4].tolist()
                scores = nms_dets[:, 4]
                classes = nms_dets[:, 5].astype(int)

            # Draw boxes only if we have detections
            if len(boxes) > 0:
                scores = scores.tolist()
                for box, score, cls in zip(boxes, scores, classes):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{COCO_NAMES[cls]} {score:.2f}"

                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw label
                    (label_w, label_h), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    cv2.rectangle(
                        frame,
                        (x1, y1 - label_h - 5),
                        (x1 + label_w, y1),
                        (0, 255, 0),
                        -1,
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                    )

        # Add info overlay
        info_text = f"Inference: {inference_time:.1f}ms | Detections: {len(boxes)}"
        cv2.putText(
            frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        cv2.putText(
            frame,
            f"FPS: {fps_display}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Show frame
        cv2.imshow("YOLOv8n INT8 - Press 'q' to quit", frame)

        # Track timing
        frame_times.append(inference_time)
        frame_count += 1

        # Calculate FPS every second
        current_time = time.time()
        if current_time - last_fps_time >= 1.0:
            fps_display = frame_count
            frame_count = 0
            last_fps_time = current_time

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Print stats
    if frame_times:
        print(f"\n=== Performance Stats ===")
        print(f"Total frames: {len(frame_times)}")
        print(f"Latency - Min: {min(frame_times):.1f}ms")
        print(f"Latency - Max: {max(frame_times):.1f}ms")
        print(f"Latency - Avg: {sum(frame_times) / len(frame_times):.1f}ms")


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    run_camera_test(model_path, conf_threshold=conf)
