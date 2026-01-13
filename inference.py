"""
This script runs inference on a video file using:
- DETRPose for skeleton extraction (17 keypoints)
- Trained ST-GCN model for action recognition

Output: Annotated video with action predictions
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from collections import deque
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Pose"))

from src.core import LazyConfig, instantiate
from Actionsrecognition.Models import TwoStreamSpatialTemporalGraph

load_dotenv()

POSE_CONFIG = os.getenv("POSE_CONFIG", "Pose/config/detrpose_hgnetv2_l.py")
POSE_WEIGHTS = os.getenv("POSE_WEIGHTS", "Pose/weights/detrpose_hgnetv2_l.pth")
POSE_THRESHOLD = float(os.getenv("POSE_THRESHOLD", 0.5))
POSE_DEVICE = os.getenv("POSE_DEVICE", "cuda")
MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", "saved/TSSTG_COCO17_20260112_172400/tsstg-model-best.pth")
INPUT_VIDEO = os.getenv("INPUT_VIDEO", "INPUT/fall_slow.mp4")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
N_FRAMES = int(os.getenv("N_FRAMES", 30))
NUM_KEYPOINTS = int(os.getenv("NUM_KEYPOINTS", 17))

# Class names (10 actions)
CLASS_NAMES = [
    "Fall backwards", "Fall forward", "Fall left", "Fall right", "Fall sitting",
    "Hop", "Kneel", "Pick up object", "Sit down", "Walk"
]

# Fall classes for highlighting
FALL_CLASSES = {0, 1, 2, 3, 4}  # Fall backwards, forward, left, right, sitting

# COCO-17 skeleton for visualization
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Face
    (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 6), (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
    (3, 5), (4, 6)  # Ears to shoulders
]

# Colors (BGR)
COLOR_NORMAL = (0, 255, 0)  # Green
COLOR_FALL = (0, 0, 255)    # Red
COLOR_SKELETON = (255, 255, 0)  # Cyan


def create_pose_model(config_path, weights_path, device):
    """Load DETRPose model."""
    print(f"Loading DETRPose model...")
    
    cfg = LazyConfig.load(config_path)
    if hasattr(cfg.model.backbone, 'pretrained'):
        cfg.model.backbone.pretrained = False
    
    model = instantiate(cfg.model)
    postprocessor = instantiate(cfg.postprocessor)
    
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
    model.load_state_dict(state)
    
    class InferenceModel(nn.Module):
        def __init__(self, model, postprocessor):
            super().__init__()
            self.model = model.deploy()
            self.postprocessor = postprocessor.deploy()
        
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    inference_model = InferenceModel(model, postprocessor).to(device)
    inference_model.eval()
    print("  DETRPose loaded!")
    return inference_model


def create_action_model(weights_path, device):
    """Load trained ST-GCN model."""
    print(f"Loading ST-GCN model from {weights_path}...")
    
    graph_args = {'layout': 'coco17', 'strategy': 'spatial'}
    model = TwoStreamSpatialTemporalGraph(graph_args, len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    print("  ST-GCN loaded!")
    return model


def normalize_keypoints(kpts, img_width, img_height):
    """Normalize keypoints to 0-1 range."""
    kpts = kpts.copy()
    kpts[:, 0] /= img_width
    kpts[:, 1] /= img_height
    return kpts


def scale_pose(xy):
    """Scale pose to -1 to 1 range."""
    xy = xy.copy()
    valid_mask = ~(np.isnan(xy).any(axis=1) | (xy == 0).all(axis=1))
    
    if valid_mask.sum() < 2:
        return xy
    
    valid_xy = xy[valid_mask]
    xy_min = np.min(valid_xy, axis=0)
    xy_max = np.max(valid_xy, axis=0)
    xy_range = xy_max - xy_min
    xy_range[xy_range == 0] = 1
    
    xy = ((xy - xy_min) / xy_range) * 2 - 1
    return xy


def draw_skeleton(frame, keypoints, color=COLOR_SKELETON, thickness=2):
    """Draw skeleton on frame."""
    for p1, p2 in COCO_SKELETON:
        if p1 < len(keypoints) and p2 < len(keypoints):
            x1, y1 = int(keypoints[p1][0]), int(keypoints[p1][1])
            x2, y2 = int(keypoints[p2][0]), int(keypoints[p2][1])
            
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Draw keypoints
    for i, (x, y) in enumerate(keypoints):
        x, y = int(x), int(y)
        if x > 0 and y > 0:
            cv2.circle(frame, (x, y), 4, color, -1)
    
    return frame


def draw_action_label(frame, action_name, action_id, confidence, position=(20, 50)):
    """Draw action label on frame."""
    is_fall = action_id in FALL_CLASSES
    color = COLOR_FALL if is_fall else COLOR_NORMAL
    
    text = f"{action_name}: {confidence:.2f}"
    if is_fall:
        text = f"[FALL] {text}"
    
    # Draw background rectangle
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    x, y = position
    cv2.rectangle(frame, (x - 5, y - text_h - 10), (x + text_w + 5, y + 5), (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    return frame


class SkeletonBuffer:
    """Buffer to store skeleton sequences for action recognition."""
    
    def __init__(self, max_frames=30, num_keypoints=17):
        self.max_frames = max_frames
        self.num_keypoints = num_keypoints
        self.buffer = deque(maxlen=max_frames)
        self.scores = deque(maxlen=max_frames)
    
    def add(self, keypoints, score):
        """Add keypoints to buffer."""
        self.buffer.append(keypoints)
        self.scores.append(score)
    
    def is_ready(self):
        """Check if buffer has enough frames."""
        return len(self.buffer) >= self.max_frames
    
    def get_sequence(self, img_width, img_height):
        """Get normalized sequence for model input."""
        if not self.is_ready():
            return None
        
        # Stack keypoints: (T, K, 2)
        kpts = np.array(list(self.buffer))
        scores = np.array(list(self.scores))
        
        # Normalize each frame
        for i in range(len(kpts)):
            kpts[i] = normalize_keypoints(kpts[i], img_width, img_height)
            kpts[i] = scale_pose(kpts[i])
        
        # Add score channel: (T, K, 3)
        scores_expanded = np.tile(scores[:, np.newaxis, np.newaxis], (1, self.num_keypoints, 1))
        kpts_with_scores = np.concatenate([kpts, scores_expanded], axis=-1)
        
        return kpts_with_scores


def predict_action(model, sequence, device):
    """Run action prediction on sequence."""
    # sequence shape: (T, K, C) -> (1, C, T, K)
    pts = torch.tensor(sequence, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    
    # Create motion input
    mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
    
    pts = pts.to(device)
    mot = mot.to(device)
    
    with torch.no_grad():
        output = model((pts, mot))
    
    probs = output[0].cpu().numpy()
    action_id = np.argmax(probs)
    confidence = probs[action_id]
    
    return action_id, confidence, probs


def process_video(input_path, output_path, pose_model, action_model, device):
    """Process video with pose estimation and action recognition."""
    
    print(f"\nProcessing video: {input_path}")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Image transform for pose model
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    
    # Skeleton buffer
    skeleton_buffer = SkeletonBuffer(max_frames=N_FRAMES, num_keypoints=NUM_KEYPOINTS)
    
    # Current action state
    current_action = "Waiting for detection..."
    current_action_id = -1
    current_confidence = 0.0
    
    frame_count = 0
    
    print("\nProcessing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Convert to PIL for pose model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        # Run pose estimation
        orig_size = torch.tensor([[width, height]]).to(device)
        im_data = transforms(frame_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = pose_model(im_data, orig_size)
        
        scores, labels, keypoints = output
        scores = scores[0].detach().cpu().numpy()
        keypoints = keypoints[0].detach().cpu().numpy()
        
        # Filter by threshold and get best detection
        valid_mask = scores > POSE_THRESHOLD
        
        if valid_mask.any():
            valid_scores = scores[valid_mask]
            valid_keypoints = keypoints[valid_mask]
            
            # Get best detection (highest score)
            best_idx = np.argmax(valid_scores)
            best_score = valid_scores[best_idx]
            best_keypoints = valid_keypoints[best_idx]  # (17, 2)
            
            # Add to buffer
            skeleton_buffer.add(best_keypoints, best_score)
            
            # Draw skeleton
            skeleton_color = COLOR_FALL if current_action_id in FALL_CLASSES else COLOR_SKELETON
            frame = draw_skeleton(frame, best_keypoints, skeleton_color)
            
            # Run action recognition if buffer is ready
            if skeleton_buffer.is_ready():
                sequence = skeleton_buffer.get_sequence(width, height)
                if sequence is not None:
                    action_id, confidence, probs = predict_action(action_model, sequence, device)
                    current_action_id = action_id
                    current_action = CLASS_NAMES[action_id]
                    current_confidence = confidence
        
        # Visualize
        frame = draw_action_label(frame, current_action, current_action_id, current_confidence)
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        out.write(frame)
        
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")

        cv2.imshow('Fall Detection', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nStopped by user")
            break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\nOutput saved to: {output_path}")


def main():
    if not os.path.exists(INPUT_VIDEO):
        print(f"Error: Input video not found: {INPUT_VIDEO}")
        return
    
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"Error: Model weights not found: {MODEL_WEIGHTS}")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Output path
    input_name = os.path.splitext(os.path.basename(INPUT_VIDEO))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{input_name}_output.mp4")
    
    print(f"\nConfiguration:")
    print(f"  Input: {INPUT_VIDEO}")
    print(f"  Output: {output_path}")
    print(f"  Model: {MODEL_WEIGHTS}")
    print(f"  Device: {POSE_DEVICE}")
    
    # Load models
    device = POSE_DEVICE
    pose_model = create_pose_model(POSE_CONFIG, POSE_WEIGHTS, device)
    action_model = create_action_model(MODEL_WEIGHTS, device)
    
    process_video(INPUT_VIDEO, output_path, pose_model, action_model, device)


if __name__ == "__main__":
    main()
