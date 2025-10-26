import streamlit as st
import cv2
import numpy as np
import tempfile
import torch
from ultralytics import YOLO
import os
import time

# --- Create Output Directory ---
os.makedirs("output", exist_ok=True)

# --- Lift Analysis Class ---
# (This class remains unchanged)
class LiftAnalysis:
    """
    Analyzes lift kinematics with improved fault detection logic,
    including persistence checks and automatic orientation detection.
    """
    def __init__(self, keypoints_data, frame_rate):
        self.keypoints_data = keypoints_data
        self.frame_rate = frame_rate if frame_rate > 0 else 30
        self.dt = 1 / self.frame_rate
        self.num_frames = len(keypoints_data)
        
        self.keypoint_map = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }
        self.orientation = self._determine_lifter_orientation()
        self._preprocess_kinematics()

    def _determine_lifter_orientation(self):
        for frame_kps in self.keypoints_data:
            if frame_kps is not None:
                left_conf = frame_kps[self.keypoint_map['left_shoulder']][2] + frame_kps[self.keypoint_map['left_hip']][2]
                right_conf = frame_kps[self.keypoint_map['right_shoulder']][2] + frame_kps[self.keypoint_map['right_hip']][2]
                if left_conf > right_conf + 0.1: return 'left'
                elif right_conf > left_conf + 0.1: return 'right'
        return 'right'

    def _get_point(self, name, frame):
        if frame < self.num_frames and self.keypoints_data[frame] is not None:
            kp = self.keypoints_data[frame][self.keypoint_map[name]]
            return kp if kp[2] > 0.1 else None
        return None

    def _calculate_angle(self, p1, p2, p3):
        if p1 is None or p2 is None or p3 is None: return None
        v1, v2 = np.array([p1[0] - p2[0], p1[1] - p2[1]]), np.array([p3[0] - p2[0], p3[1] - p2[1]])
        angle = abs(np.degrees(np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])))
        return angle if angle <= 180 else 360 - angle

    def _get_bar_position(self, frame):
        lw, rw = (self._get_point(n, frame) for n in ['left_wrist', 'right_wrist'])
        if lw is not None and rw is not None: return np.mean([lw[:2], rw[:2]], axis=0)
        return None

    def _preprocess_kinematics(self):
        self.bar_y = [p[1] if p is not None else None for p in [self._get_bar_position(i) for i in range(self.num_frames)]]
        self.hip_angles, self.elbow_angles = [], []
        sh_name, hip_name, knee_name, elb_name, wri_name = (f"{self.orientation}_{n}" for n in ['shoulder', 'hip', 'knee', 'elbow', 'wrist'])
        for i in range(self.num_frames):
            sh, hip, knee, elb, wri = (self._get_point(n, i) for n in [sh_name, hip_name, knee_name, elb_name, wri_name])
            self.hip_angles.append(self._calculate_angle(sh, hip, knee))
            self.elbow_angles.append(self._calculate_angle(sh, elb, wri))

    def analyze_lift(self):
        feedback = []
        valid_bar_y = [y for y in self.bar_y if y is not None]
        if not valid_bar_y: return {"feedback": ["Could not detect barbell path."], "verdict": "Bad Lift", "start_frame": None, "end_of_pull_frame": None}
        
        floor_y = np.max(valid_bar_y)
        try: start_frame = next(i for i, y in enumerate(self.bar_y) if y is not None and y < floor_y - 10)
        except StopIteration: return {"feedback": ["Could not detect lift start."], "verdict": "Bad Lift", "start_frame": None, "end_of_pull_frame": None}

        clean_bar_y_pull = [y if y is not None else float('inf') for y in self.bar_y[start_frame:]]
        if not clean_bar_y_pull: return {"feedback": ["Analysis failed after start."], "verdict": "Bad Lift", "start_frame": start_frame, "end_of_pull_frame": None}
        
        end_of_pull_frame = np.argmin(clean_bar_y_pull) + start_frame
        
        pull_phase_hip_angles = self.hip_angles[start_frame:end_of_pull_frame + 1]
        valid_pull_hip_angles = [a for a in pull_phase_hip_angles if a is not None]

        if not valid_pull_hip_angles: return {"feedback": ["Could not analyze hip extension."], "verdict": "Bad Lift", "start_frame": start_frame, "end_of_pull_frame": end_of_pull_frame}
        
        peak_hip_angle = np.max(valid_pull_hip_angles)
        if peak_hip_angle < 170: feedback.append(f"Incomplete Hip Extension (Peak: {int(peak_hip_angle)}deg)")

        peak_hip_angle_index_in_pull = np.argmax([a if a is not None else -1 for a in pull_phase_hip_angles])
        
        bent_arm_counter, persistence_threshold = 0, 3
        for i in range(peak_hip_angle_index_in_pull):
            elbow_angle = self.elbow_angles[start_frame + i]
            if elbow_angle is not None and elbow_angle < 160: bent_arm_counter += 1
            else: bent_arm_counter = 0
            if bent_arm_counter >= persistence_threshold:
                feedback.append("Early Arm Bend detected before full extension.")
                break

        verdict = "Good Lift" if not feedback else "Bad Lift"
        if not feedback: feedback.append("No major technical faults detected during the pull.")
            
        return {"feedback": feedback, "verdict": verdict, "start_frame": start_frame, "end_of_pull_frame": end_of_pull_frame}

# --- Drawing Utilities & App ---
def draw_feedback_on_frame(frame, verdict, feedback_list):
    verdict_color = (0, 255, 0) if verdict == "Good Lift" else (0, 0, 255)
    cv2.putText(frame, verdict, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, verdict_color, 3, cv2.LINE_AA)
    y_pos = 90
    for text in feedback_list:
        cv2.putText(frame, text, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y_pos += 30
    return frame

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1, area2 = (box1[2] - box1[0]) * (box1[3] - box1[1]), (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

st.set_page_config(page_title="LiftCoach AI", layout="wide")
st.title("🏋️ LiftCoach AI - Athlete Tracking & Analysis")
st.write("Upload a video to receive a detailed analysis. The AI will lock onto the main athlete.")

@st.cache_resource
def load_model():
    return YOLO('yolov8n-pose.pt')

model = load_model()

st.sidebar.header("Input Options")
uploaded_file = st.sidebar.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])

if uploaded_file:
    process_button = st.sidebar.button("Analyze Lift", key="process")
    if process_button:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        cap = None
        writer = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): raise Exception("Error: Could not open video file.")

            st.info("Phase 1: Identifying & Tracking Athlete...")
            progress_bar = st.progress(0, text="Analyzing Frames...")
            
            all_keypoints, annotated_frames, target_bbox = [], [], None
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret: break
                
                results = model.predict(frame, verbose=False)
                annotated_frames.append(results[0].plot())
                
                detections = [{'box': box.xyxy[0].cpu().numpy(), 'kps': kps.data[0].cpu().numpy()} 
                              for box, kps in zip(results[0].boxes, results[0].keypoints) if box.conf[0] > 0.5]
                
                if not detections:
                    all_keypoints.append(None); continue

                if target_bbox is None:
                    target = max(detections, key=lambda d: (d['box'][2]-d['box'][0])*(d['box'][3]-d['box'][1]))
                    target_bbox = target['box']
                    all_keypoints.append(target['kps'])
                else:
                    best_match, max_iou = None, 0.3
                    for det in detections:
                        iou = calculate_iou(target_bbox, det['box'])
                        if iou > max_iou: max_iou, best_match = iou, det
                    
                    if best_match:
                        target_bbox = best_match['box']
                        all_keypoints.append(best_match['kps'])
                    else:
                        all_keypoints.append(None)
                
                progress_bar.progress((i + 1) / total_frames, text=f"Tracking Frame {i+1}/{total_frames}")

            st.info("Phase 2: Analyzing lift mechanics...")
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            analyzer = LiftAnalysis(all_keypoints, frame_rate)
            analysis_results = analyzer.analyze_lift()
            
            st.info("Phase 3: Generating final video file...")
            output_filename, output_path = f"analyzed_{int(time.time())}.mp4", os.path.join("output", f"analyzed_{int(time.time())}.mp4")
            frame_h, frame_w, _ = annotated_frames[0].shape
            
            # --- THE FIX IS HERE ---
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Changed from 'avc1' to 'mp4v'
            writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_w, frame_h))

            for i, frame in enumerate(annotated_frames):
                frame = draw_feedback_on_frame(frame, analysis_results['verdict'], []) # Simplified feedback on video
                if i == analysis_results.get('start_frame'): cv2.putText(frame, "LIFT START", (frame_w - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                if i == analysis_results.get('end_of_pull_frame'): cv2.putText(frame, "END OF PULL", (frame_w - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            writer.release()
            writer = None

            st.success(f"Analysis complete! Preview below.")
            with open(output_path, 'rb') as f: video_bytes = f.read()
            st.video(video_bytes)
            st.download_button(label="Download Analyzed Video", data=video_bytes, file_name=output_filename, mime="video/mp4")

        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            if cap: cap.release()
            if writer: writer.release()
            if 'video_path' in locals() and os.path.exists(video_path):
                try: os.remove(video_path)
                except Exception: pass

st.sidebar.info(
    "**Disclaimer:** This tool is for educational purposes. Feedback is AI-generated and should not replace advice from a qualified human coach."
)
