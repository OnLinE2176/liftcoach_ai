import streamlit as st
import cv2
import numpy as np
import tempfile
import torch
from ultralytics import YOLO
import os
import time
import json
import matplotlib.pyplot as plt

# --- Create Output Directory ---
os.makedirs("output", exist_ok=True)

# --- Lift Analysis Class - Aligned with Thesis Objectives ---
class LiftAnalysis:
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
                left_conf, right_conf = sum(frame_kps[self.keypoint_map[n]][2] for n in ['left_shoulder', 'left_hip']), sum(frame_kps[self.keypoint_map[n]][2] for n in ['right_shoulder', 'right_hip'])
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
        self.bar_positions = [self._get_bar_position(i) for i in range(self.num_frames)]
        self.bar_x = [p[0] if p is not None else None for p in self.bar_positions]
        self.bar_y = [p[1] if p is not None else None for p in self.bar_positions]
        
        self.hip_angles, self.elbow_angles = [], []
        sh_name, hip_name, knee_name, elb_name, wri_name = (f"{self.orientation}_{n}" for n in ['shoulder', 'hip', 'knee', 'elbow', 'wrist'])
        for i in range(self.num_frames):
            sh, hip, knee, elb, wri = (self._get_point(n, i) for n in [sh_name, hip_name, knee_name, elb_name, wri_name])
            self.hip_angles.append(self._calculate_angle(sh, hip, knee))
            self.elbow_angles.append(self._calculate_angle(sh, elb, wri))

    def analyze_lift(self):
        faults_found, kinematic_data = [], {}
        
        valid_bar_y = [y for y in self.bar_y if y is not None]
        if not valid_bar_y:
            return {"faults_found": ["Could not detect barbell path."], "verdict": "Bad Lift", "phases": {}, "kinematic_data": {}}
        
        floor_y = np.max(valid_bar_y)
        try: start_frame = next(i for i, y in enumerate(self.bar_y) if y is not None and y < floor_y - 10)
        except StopIteration:
            return {"faults_found": ["Could not detect lift start."], "verdict": "Bad Lift", "phases": {}, "kinematic_data": {}}

        clean_bar_y_pull = [y if y is not None else float('inf') for y in self.bar_y[start_frame:]]
        if not clean_bar_y_pull:
            return {"faults_found": ["Analysis failed after start."], "verdict": "Bad Lift", "phases": {"start_frame": start_frame}, "kinematic_data": {}}
        
        end_of_pull_frame = np.argmin(clean_bar_y_pull) + start_frame
        phases = {"start_frame": start_frame, "end_of_pull_frame": end_of_pull_frame}

        # --- Fault Check 1: Incomplete Hip Extension ---
        pull_phase_hip_angles = self.hip_angles[start_frame:end_of_pull_frame + 1]
        valid_pull_hip_angles = [a for a in pull_phase_hip_angles if a is not None]
        if valid_pull_hip_angles:
            peak_hip_angle = np.max(valid_pull_hip_angles)
            kinematic_data['peak_hip_angle'] = round(peak_hip_angle, 2)
            if peak_hip_angle < 170: faults_found.append("Incomplete Hip Extension")
        
        # --- Fault Check 2: Early Arm Bend ---
        if valid_pull_hip_angles:
            peak_hip_angle_index_in_pull = np.argmax([a if a is not None else -1 for a in pull_phase_hip_angles])
            bent_arm_counter, persistence_threshold = 0, 3
            for i in range(peak_hip_angle_index_in_pull):
                elbow_angle = self.elbow_angles[start_frame + i]
                if elbow_angle is not None and elbow_angle < 160: bent_arm_counter += 1
                else: bent_arm_counter = 0
                if bent_arm_counter >= persistence_threshold:
                    faults_found.append("Early Arm Bend")
                    kinematic_data['early_arm_bend_frame'] = start_frame + i
                    break
        
        # --- NEW (Thesis Feature): Fault Check 3: Bar Forward in Catch ---
        hip_y = [p[1] if p is not None else None for p in [self._get_point(f"{self.orientation}_hip", i) for i in range(self.num_frames)]]
        hip_y_after_pull = [y for y in hip_y[end_of_pull_frame:] if y is not None]
        if hip_y_after_pull:
            clean_hip_y_after_pull = [y if y is not None else -1 for y in hip_y[end_of_pull_frame:]]
            catch_frame = np.argmax(clean_hip_y_after_pull) + end_of_pull_frame
            phases['catch_frame'] = catch_frame
            
            bar_x_start = self.bar_x[start_frame]
            bar_x_catch = self.bar_x[catch_frame]
            frame_width = self.keypoints_data[0].shape[1] # Use keypoint coordinate system width
            
            if bar_x_start is not None and bar_x_catch is not None:
                horizontal_deviation = bar_x_catch - bar_x_start
                kinematic_data['horizontal_bar_deviation_pixels'] = round(horizontal_deviation, 2)
                # If bar moves forward by more than 5% of the frame width, flag it
                if horizontal_deviation > (0.05 * frame_width):
                    faults_found.append("Bar Forward in Catch")
        
        verdict = "Good Lift" if not faults_found else "Bad Lift"
        return {"faults_found": faults_found, "verdict": verdict, "phases": phases, "kinematic_data": kinematic_data, "bar_path": self.bar_positions}

# --- Drawing Utilities & App ---
def draw_feedback_on_frame(frame, verdict):
    verdict_color = (0, 255, 0) if verdict == "Good Lift" else (0, 0, 255)
    cv2.putText(frame, verdict, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, verdict_color, 3, cv2.LINE_AA)
    return frame

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1, area2 = (box1[2] - box1[0]) * (box1[3] - box1[1]), (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

st.set_page_config(page_title="LiftCoach AI", layout="wide")
st.title("🏋️ LiftCoach AI - Thesis Implementation")
st.write("A computer vision tool for analyzing Olympic Weightlifting technique, aligned with the ITS200-2 project plan.")

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

        cap, writer = None, None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): raise Exception("Error: Could not open video file.")

            st.info("Phase 1: Tracking athlete and analyzing frames...")
            # (The tracking loop remains unchanged)
            progress_bar, all_keypoints, raw_frames, target_bbox = st.progress(0, text="Analyzing Frames..."), [], [], None
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for i in range(total_frames):
                ret, frame = cap.read();
                if not ret: break
                raw_frames.append(frame)
                results = model.predict(frame, verbose=False)
                detections = [{'box': box.xyxy[0].cpu().numpy(), 'kps': kps.data[0].cpu().numpy()} for box, kps in zip(results[0].boxes, results[0].keypoints) if box.conf[0] > 0.5]
                if not detections: all_keypoints.append(None); continue
                if target_bbox is None:
                    target = max(detections, key=lambda d: (d['box'][2]-d['box'][0])*(d['box'][3]-d['box'][1])); target_bbox = target['box']; all_keypoints.append(target['kps'])
                else:
                    best_match, max_iou = None, 0.3
                    for det in detections:
                        iou = calculate_iou(target_bbox, det['box']);
                        if iou > max_iou: max_iou, best_match = iou, det
                    if best_match: target_bbox = best_match['box']; all_keypoints.append(best_match['kps'])
                    else: all_keypoints.append(None)
                progress_bar.progress((i + 1) / total_frames, text=f"Analyzing Frame {i+1}/{total_frames}")

            st.info("Phase 2: Analyzing lift mechanics...")
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            analyzer = LiftAnalysis(all_keypoints, frame_rate)
            analysis_results = analyzer.analyze_lift()
            
            st.success("Analysis Complete!")
            
            # --- NEW: Dashboard Display ---
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Analysis Dashboard")
                st.metric(label="Final Verdict", value=analysis_results['verdict'])
                st.subheader("Detected Faults")
                if not analysis_results['faults_found'] or analysis_results['verdict'] == "Good Lift":
                    st.write("✅ No major technical faults detected.")
                else:
                    for fault in analysis_results['faults_found']:
                        st.warning(f"⚠️ {fault}")
                with st.expander("Show Raw Diagnostic Data (JSON)"):
                    st.json(analysis_results)

            with col2:
                st.subheader("Barbell Path Visualization")
                bar_path_data = analysis_results.get("bar_path")
                if bar_path_data:
                    x_coords = [p[0] for p in bar_path_data if p is not None]
                    y_coords = [p[1] for p in bar_path_data if p is not None]
                    if x_coords and y_coords:
                        fig, ax = plt.subplots()
                        ax.plot(x_coords, y_coords, marker='o', linestyle='-', color='cyan')
                        ax.set_title("Barbell Trajectory")
                        ax.set_xlabel("Horizontal Position")
                        ax.set_ylabel("Vertical Position")
                        ax.invert_yaxis() # Invert y-axis so top of screen is "up"
                        ax.grid(True)
                        ax.set_aspect('equal', adjustable='box')
                        st.pyplot(fig)

            st.divider()
            st.subheader("Analyzed Video Output")
            with st.spinner("Generating final video file..."):
                output_filename, output_path = f"analyzed_{int(time.time())}.mp4", os.path.join("output", f"analyzed_{int(time.time())}.mp4")
                frame_h, frame_w, _ = raw_frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_w, frame_h))
                
                # Re-run predict to get annotated frames for the target only
                for i, frame in enumerate(raw_frames):
                    kps = all_keypoints[i]
                    if kps is not None:
                        # Create a dummy results object with only the tracked person
                        results = model.predict(frame, verbose=False) # Get a results object shell
                        results[0].keypoints = YOLO(model.ckpt_path).predictor.postprocess_pose([torch.from_numpy(kps).unsqueeze(0)], frame, frame)[0].keypoints
                        annotated_frame = results[0].plot()
                    else:
                        annotated_frame = frame # Use raw frame if no one was tracked
                    
                    annotated_frame = draw_feedback_on_frame(annotated_frame, analysis_results['verdict'])
                    phases = analysis_results.get('phases', {})
                    if i == phases.get('start_frame'): cv2.putText(annotated_frame, "LIFT START", (frame_w - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    if i == phases.get('end_of_pull_frame'): cv2.putText(annotated_frame, "END OF PULL", (frame_w - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    if i == phases.get('catch_frame'): cv2.putText(annotated_frame, "CATCH", (frame_w - 300, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    
                    writer.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

                writer.release()
                writer = None

            with open(output_path, 'rb') as f:
                video_bytes = f.read()
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
