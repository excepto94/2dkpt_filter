import cv2
import torch
import numpy as np
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchvision.transforms.functional import to_tensor

video_path = "35000016_20250904_08-57-16_spa3_oms_logfile_video.ts"


keypoints_path = "keypoints_data.npz"
updated_keypoints_path = "updated_keypoints_data.npz"
max_frames = 6500

def run_network():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    weights = KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
    model = keypointrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file")

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    x1, y1 = 0, 0
    x2, y2 = width, height
    frame_keypoints = []

    with torch.no_grad():
        for frame_idx in range(max_frames):
            ret, frame = cap.read()

            # Crop lower-left quadrant
            roi = frame[y1:y2, x1:x2]

            # Convert to tensor and move to device
            img_tensor = to_tensor(roi).unsqueeze(0).to(device)

            # Inference
            outputs = model(img_tensor)[0]

            # Collect keypoints for this frame
            frame_kps = []
            for i, score in enumerate(outputs["scores"]):
                if score < 0.8:
                    continue
                kps = outputs["keypoints"][i].detach().cpu().numpy()  # shape (17, 3)
                frame_kps.append(kps)

            frame_keypoints.append(frame_kps)

            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx+1}/{max_frames} frames...")

    cap.release()

    # Save keypoints as numpy file
    np.savez_compressed(
        keypoints_path,
        keypoints=np.array(frame_keypoints, dtype=object)
    )
    print(f"Saved keypoints for {len(frame_keypoints)} frames → {keypoints_path}")

def update_keypoints_with_history():
    # Number of past frames to include
    history = 60

    # Load original keypoints
    data = np.load(keypoints_path, allow_pickle=True)
    frame_keypoints = data["keypoints"]  # list of [persons][keypoints][3]

    updated_keypoints = []

    for i in range(len(frame_keypoints)):
        # Collect the keypoints for this frame and the previous `history - 1` frames
        start_idx = max(0, i - history + 1)
        past_frames = frame_keypoints[start_idx:i + 1]

        combined = []
        for f in past_frames:
            combined.extend(f)  # merge all people across these frames

        updated_keypoints.append(combined)

    # Convert to an object array (to handle variable-length frames)
    updated_keypoints = np.array(updated_keypoints, dtype=object)

    # Save safely
    np.savez_compressed(updated_keypoints_path, keypoints=updated_keypoints)
    print(f"Saved updated keypoints with {history}-frame trails → {updated_keypoints_path}")

def create_video():
    # Load keypoints
    data = np.load(updated_keypoints_path, allow_pickle=True)
    frame_keypoints = data["keypoints"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file")

    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    x1, y1 = 0, 0
    x2, y2 = width, height

    out = cv2.VideoWriter("keypoints_overlay.mp4",
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps, (width, height))

    for frame_idx in range(max_frames):
        ret, frame = cap.read()

        # Crop lower-left quadrant
        roi = frame[y1:y2, x1:x2]

        for person_kps in frame_keypoints[frame_idx]:
            for (x, y, v) in person_kps:
                if v > 0:  # visible
                    cv2.circle(roi, (int(x), int(y)), 3, (0, 255, 0), -1)

        cv2.imshow("Keypoint Overlay", roi)
        out.write(roi)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Saved overlay video → keypoints_overlay.mp4")


if __name__ == "__main__":
    run_network()
    update_keypoints_with_history()
    create_video()
