import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
import os
import time
import json
import matplotlib.pyplot as plt

# --- Lift Analysis Class (Thesis Version) ---
class LiftAnalysis:
    """
    Analyzes lift kinematics, detects IWF rule violations, and generates 
    structured diagnostic data for visualization.
    """
    def __init__(self, keypoints_data, frame_rate):
        self.keypoints_data = keypoints_data
        self.frame_rate = frame_rate if frame_rate > 0 else 30
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
                left_conf = sum(frame_kps[self.keypoint_map[n]][2] for n in ['left_shoulder', 'left_hip'])
                right_conf = sum(frame_kps[self.keypoint_map[n]][2] for n in ['right_shoulder', 'right_hip'])
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
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        angle = abs(np.degrees(np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])))
        return angle if angle <= 180 else 360 - angle

    def _get_bar_position(self, frame):
        lw = self._get_point('left_wrist', frame)
        rw = self._get_point('right_wrist', frame)
        if lw is not None and rw is not None:
            return np.mean([lw[:2], rw[:2]], axis=0)
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
        faults_found, kinematic_data, phases = [], {}, {}
        valid_bar_y = [y for y in self.bar_y if y is not None]
        if not valid_bar_y: return {"faults_found": ["Could not detect barbell path."], "verdict": "Bad Lift", "phases": {}, "kinematic_data": {}, "bar_path": []}
        
        floor_y = np.max(valid_bar_y)
        try: start_frame = next(i for i, y in enumerate(self.bar_y) if y is not None and y < floor_y - 10)
        except StopIteration: return {"faults_found": ["Could not detect lift start."], "verdict": "Bad Lift", "phases": {}, "kinematic_data": {}, "bar_path": []}

        clean_bar_y_pull = [y if y is not None else float('inf') for y in self.bar_y[start_frame:]]
        if not clean_bar_y_pull: return {"faults_found": ["Analysis failed after start."], "verdict": "Bad Lift", "phases": {"start_frame": start_frame}, "kinematic_data": {}, "bar_path": []}
        
        end_of_pull_frame = np.argmin(clean_bar_y_pull) + start_frame
        phases = {"start_frame": start_frame, "end_of_pull_frame": end_of_pull_frame}

        # 1. Incomplete Extension
        pull_hip_angles = [a for a in self.hip_angles[start_frame:end_of_pull_frame+1] if a is not None]
        if pull_hip_angles:
            peak_hip = np.max(pull_hip_angles)
            kinematic_data['peak_hip_angle'] = round(peak_hip, 2)
            if peak_hip < 170: faults_found.append("Incomplete Hip Extension")
        
        # 2. Early Arm Bend
        if pull_hip_angles:
            peak_hip_idx = np.argmax([a if a is not None else -1 for a in self.hip_angles[start_frame:end_of_pull_frame+1]])
            bent_count = 0
            for i in range(peak_hip_idx):
                angle = self.elbow_angles[start_frame + i]
                if angle is not None and angle < 160: bent_count += 1
                else: bent_count = 0
                if bent_count >= 3:
                    faults_found.append("Early Arm Bend")
                    kinematic_data['early_arm_bend_frame'] = start_frame + i
                    break

        # 3. Bar Forward in Catch (Thesis Feature)
        hip_y = [p[1] if p is not None else None for p in [self._get_point(f"{self.orientation}_hip", i) for i in range(self.num_frames)]]
        hip_y_catch = [y if y is not None else -1 for y in hip_y[end_of_pull_frame:]]
        if hip_y_catch:
            catch_frame = np.argmax(hip_y_catch) + end_of_pull_frame
            phases['catch_frame'] = catch_frame
            
            # Safe frame width check
            frame_width = next((kps.shape[1] for kps in self.keypoints_data if kps is not None), None)
            
            if self.bar_x[start_frame] and self.bar_x[catch_frame] and frame_width:
                 dev = self.bar_x[catch_frame] - self.bar_x[start_frame]
                 kinematic_data['bar_deviation_px'] = round(dev, 2)
                 # If bar is >5% of screen width forward from start position
                 if (self.orientation == 'right' and dev > frame_width * 0.05) or \
                    (self.orientation == 'left' and dev < -frame_width * 0.05):
                     faults_found.append("Bar Forward in Catch")

        return {"faults_found": faults_found, "verdict": "Good Lift" if not faults_found else "Bad Lift", "phases": phases, "kinematic_data": kinematic_data, "bar_path": self.bar_positions}

# --- Drawing Utilities ---
def draw_feedback(frame, verdict):
    color = (0, 255, 0) if verdict == "Good Lift" else (0, 0, 255)
    cv2.putText(frame, verdict, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)
    return frame

def get_iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return inter / union if union > 0 else 0

# --- Streamlit App ---
st.set_page_config(page_title="LiftCoach AI", layout="wide")
st.title("🏋️ LiftCoach AI - Thesis Implementation")
st.write("Automated technical analysis for Olympic Weightlifting based on IWF standards.")

@st.cache_resource
def load_model(): return YOLO('yolov8n-pose.pt')
model = load_model()

st.sidebar.header("Video Input")
uploaded_file = st.sidebar.file_uploader("Upload a lift video (MP4/MOV)", type=["mp4", "mov", "avi"])

if uploaded_file and st.sidebar.button("Analyze Lift"):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise Exception("Error opening video file.")

        st.info("Phase 1/3: Tracking athlete & extracting kinematics...")
        prog_bar = st.progress(0)
        all_kps, raw_frames, target_box = [], [], None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            raw_frames.append(frame)
            results = model.predict(frame, verbose=False)
            
            dets = [{'box': b.xyxy[0].cpu().numpy(), 'kps': k.data[0].cpu().numpy()} 
                    for b, k in zip(results[0].boxes, results[0].keypoints) if b.conf[0] > 0.5]
            
            if not dets: all_kps.append(None)
            elif target_box is None:
                target = max(dets, key=lambda d: (d['box'][2]-d['box'][0])*(d['box'][3]-d['box'][1]))
                target_box, all_kps.append(target['kps'])
            else:
                best = max(dets, key=lambda d: get_iou(target_box, d['box']))
                if get_iou(target_box, best['box']) > 0.3:
                    target_box, all_kps.append(best['kps'])
                else: all_kps.append(None)
            prog_bar.progress((i + 1) / total_frames)

        st.info("Phase 2/3: Running biomechanical analysis...")
        fps = cap.get(cv2.CAP_PROP_FPS)
        res = LiftAnalysis(all_kps, fps).analyze_lift()
        
        st.success("Analysis Complete!")
        
        # --- Dashboard ---
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Diagnostic Results")
            st.metric("Verdict", res['verdict'])
            if not res['faults_found'] or res['verdict'] == "Good Lift":
                st.success("✅ No major technical faults detected.")
            else:
                for f in res['faults_found']: st.error(f"⚠️ {f}")
            with st.expander("Raw Data (JSON)"): st.json(res)

        with c2:
            st.subheader("Barbell Trajectory")
            if res['bar_path']:
                valid_p = [p for p in res['bar_path'] if p is not None]
                if valid_p:
                    fig, ax = plt.subplots()
                    ax.plot([p[0] for p in valid_p], [p[1] for p in valid_p], 'c-o', markersize=2)
                    ax.invert_yaxis(); ax.set_aspect('equal'); ax.axis('off')
                    fig.patch.set_facecolor('black')
                    st.pyplot(fig)

        # --- Video Generation (Cloud Safe Mode) ---
        st.divider()
        st.subheader("Visual Feedback")
        with st.spinner("Rendering final video..."):
            temp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            h, w = raw_frames[0].shape[:2]
            # Try avc1 first (requires packages.txt on cloud), fallback to mp4v if needed locally
            fourcc = cv2.VideoWriter_fourcc(*'avc1') 
            out = cv2.VideoWriter(temp_out, fourcc, fps, (w, h))
            
            for i, frame in enumerate(raw_frames):
                if all_kps[i] is not None:
                    # Draw skeleton only for target athlete
                    for k in all_kps[i]:
                         if k[2] > 0.1: cv2.circle(frame, (int(k[0]), int(k[1])), 3, (0,255,255), -1)
                
                frame = draw_feedback(frame, res['verdict'])
                p = res['phases']
                if i == p.get('start_frame'): cv2.putText(frame, "START", (w-200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                if i == p.get('end_of_pull_frame'): cv2.putText(frame, "PEAK", (w-200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
                if i == p.get('catch_frame'): cv2.putText(frame, "CATCH", (w-200, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
                out.write(frame)
            
            out.release()
            with open(temp_out, 'rb') as f: video_bytes = f.read()
            os.remove(temp_out)

        st.video(video_bytes)
        st.download_button("Download Video", video_bytes, file_name="lift_analysis.mp4", mime="video/mp4")

    except Exception as e: st.error(f"Error: {e}")
    finally:
        if cap: cap.release()
        try: os.remove(video_path)
        except: pass
