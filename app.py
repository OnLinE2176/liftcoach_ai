import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
import os
import time
import json 
import matplotlib.pyplot as plt
from io import BytesIO

# --- Lift Analysis Class (Unchanged and Correct) ---
class LiftAnalysis:
    # ... (Keep the entire LiftAnalysis class exactly as it was. It's working correctly.)
    def __init__(self, keypoints_data, frame_rate):
        self.keypoints_data = keypoints_data
        self.frame_rate = frame_rate if frame_rate > 0 else 30
        self.dt = 1 / self.frame_rate
        self.num_frames = len(keypoints_data)
        self.keypoint_map = {'nose':0,'left_eye':1,'right_eye':2,'left_ear':3,'right_ear':4,'left_shoulder':5,'right_shoulder':6,'left_elbow':7,'right_elbow':8,'left_wrist':9,'right_wrist':10,'left_hip':11,'right_hip':12,'left_knee':13,'right_knee':14,'left_ankle':15,'right_ankle':16}
        self.orientation = self._determine_lifter_orientation()
        self._preprocess_kinematics()
    def _determine_lifter_orientation(self):
        for kps in self.keypoints_data:
            if kps is not None:
                l, r = sum(kps[self.keypoint_map[n]][2] for n in ['left_shoulder','left_hip']), sum(kps[self.keypoint_map[n]][2] for n in ['right_shoulder','right_hip'])
                if l > r + 0.1: return 'left'
                elif r > l + 0.1: return 'right'
        return 'right'
    def _get_point(self, name, frame):
        if frame < self.num_frames and self.keypoints_data[frame] is not None:
            kp = self.keypoints_data[frame][self.keypoint_map[name]]; return kp if kp[2] > 0.1 else None
        return None
    def _calculate_angle(self, p1, p2, p3):
        if p1 is None or p2 is None or p3 is None: return None
        v1, v2 = np.array([p1[0] - p2[0], p1[1] - p2[1]]), np.array([p3[0] - p2[0], p3[1] - p2[1]])
        angle = abs(np.degrees(np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0]))); return angle if angle <= 180 else 360 - angle
    def _get_bar_position(self, frame):
        lw, rw = (self._get_point(n, frame) for n in ['left_wrist', 'right_wrist']); return np.mean([lw[:2], rw[:2]], axis=0) if lw is not None and rw is not None else None
    def _preprocess_kinematics(self):
        self.bar_positions = [self._get_bar_position(i) for i in range(self.num_frames)]
        self.bar_x = [p[0] if p is not None else None for p in self.bar_positions]; self.bar_y = [p[1] if p is not None else None for p in self.bar_positions]
        self.hip_angles, self.elbow_angles = [], []
        sh_name, hip_name, knee_name, elb_name, wri_name = (f"{self.orientation}_{n}" for n in ['shoulder','hip','knee','elbow','wrist'])
        for i in range(self.num_frames):
            sh, hip, knee, elb, wri = (self._get_point(n, i) for n in [sh_name, hip_name, knee_name, elb_name, wri_name])
            self.hip_angles.append(self._calculate_angle(sh, hip, knee)); self.elbow_angles.append(self._calculate_angle(sh, elb, wri))
    def analyze_lift(self):
        faults, kin_data, phases = [], {}, {}
        valid_y = [y for y in self.bar_y if y is not None]
        if not valid_y: return {"faults_found": ["Could not detect barbell path."], "verdict": "Bad Lift", "phases": {}, "kinematic_data": {}, "bar_path": []}
        try: start_frame = next(i for i, y in enumerate(self.bar_y) if y is not None and y < np.max(valid_y) - 10)
        except StopIteration: return {"faults_found": ["Could not detect lift start."], "verdict": "Bad Lift", "phases": {}, "kinematic_data": {}, "bar_path": []}
        clean_pull = [y if y is not None else float('inf') for y in self.bar_y[start_frame:]]
        if not clean_pull: return {"faults_found": ["Analysis failed after start."], "verdict": "Bad Lift", "phases": {"start_frame": start_frame}, "kinematic_data": {}, "bar_path": []}
        end_pull = np.argmin(clean_pull) + start_frame
        phases = {"start_frame": start_frame, "end_of_pull_frame": end_pull}
        pull_hips = [a for a in self.hip_angles[start_frame:end_pull+1] if a is not None]
        if pull_hips:
            peak_hip = np.max(pull_hips); kin_data['peak_hip_angle'] = round(peak_hip, 2)
            if peak_hip < 170: faults.append("Incomplete Hip Extension")
            peak_hip_idx = np.argmax([a if a is not None else -1 for a in self.hip_angles[start_frame:end_pull+1]])
            bent_count = 0
            for i in range(peak_hip_idx):
                angle = self.elbow_angles[start_frame+i]
                if angle is not None and angle < 160: bent_count += 1
                else: bent_count = 0
                if bent_count >= 3: faults.append("Early Arm Bend"); kin_data['early_arm_bend_frame'] = start_frame+i; break
        hip_y_catch = [y for y in [self._get_point(f"{self.orientation}_hip", i)[1] for i in range(end_pull, self.num_frames) if self._get_point(f"{self.orientation}_hip", i)] if y is not None]
        if hip_y_catch:
            catch_frame = np.argmax(hip_y_catch) + end_pull; phases['catch_frame'] = catch_frame
            w = next((kps.shape[1] for kps in self.keypoints_data if kps is not None), None)
            if self.bar_x[start_frame] and self.bar_x[catch_frame] and w:
                 dev = self.bar_x[catch_frame] - self.bar_x[start_frame]; kin_data['bar_deviation_px'] = round(dev, 2)
                 if (self.orientation=='right' and dev>w*0.05) or (self.orientation=='left' and dev<-w*0.05): faults.append("Bar Forward in Catch")
        return {"faults_found": faults, "verdict": "Good Lift" if not faults else "Bad Lift", "phases": phases, "kinematic_data": kin_data, "bar_path": self.bar_positions}

# --- Drawing & App Logic ---
st.set_page_config(page_title="LiftCoach AI", layout="wide")
st.title("🏋️ LiftCoach AI - Thesis Implementation")
st.write("A computer vision tool for analyzing Olympic Weightlifting technique.")
@st.cache_resource
def load_model(): return YOLO('yolov8n-pose.pt')
model = load_model()
st.sidebar.header("Video Input")
uploaded_file = st.sidebar.file_uploader("Upload a lift video (MP4/MOV)", type=["mp4", "mov", "avi"])

def get_iou(box1, box2): # Moved helper function here
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return inter / union if union > 0 else 0

if uploaded_file and st.sidebar.button("Analyze Lift"):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4'); tfile.write(uploaded_file.read()); video_path = tfile.name
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise Exception("Error opening video file.")
        st.info("Phase 1: Analyzing frames...")
        prog_bar = st.progress(0, text="Analyzing Frames...")
        all_kps, raw_frames, target_box = [], [], None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(total_frames):
            ret, frame = cap.read();
            if not ret: break
            raw_frames.append(frame)
            results = model.predict(frame, verbose=False)
            dets = [{'box':b.xyxy[0].cpu().numpy(),'kps':k.data[0].cpu().numpy()} for b,k in zip(results[0].boxes,results[0].keypoints) if b.conf[0]>0.5]
            if not dets: all_kps.append(None); continue
            if target_box is None: target=max(dets,key=lambda d:(d['box'][2]-d['box'][0])*(d['box'][3]-d['box'][1])); target_box=target['box']; all_kps.append(target['kps'])
            else:
                best=max(dets,key=lambda d:get_iou(target_box,d['box']))
                if get_iou(target_box,best['box'])>0.3: target_box=best['box']; all_kps.append(best['kps'])
                else: all_kps.append(None)
            prog_bar.progress((i+1)/total_frames, text=f"Analyzing Frame {i+1}/{total_frames}")
        
        st.info("Phase 2: Finalizing analysis...")
        res = LiftAnalysis(all_kps, cap.get(cv2.CAP_PROP_FPS)).analyze_lift()
        st.success("Analysis Complete!")
        
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Diagnostic Results"); st.metric("Verdict",res['verdict'])
            st.subheader("Detected Faults")
            if not res['faults_found'] or res['verdict']=="Good Lift": st.success("✅ No major technical faults detected.")
            else:
                for f in res['faults_found']: st.error(f"⚠️ {f}")
            with st.expander("Raw Data (JSON)"): st.json(res)
        with c2:
            st.subheader("Barbell Trajectory"); valid_p=[p for p in res['bar_path'] if p is not None]
            if valid_p:
                fig,ax=plt.subplots(); ax.plot([p[0] for p in valid_p],[p[1] for p in valid_p],'c-o',markersize=2)
                ax.invert_yaxis(); ax.set_aspect('equal'); ax.axis('off'); fig.patch.set_facecolor('#0E1117'); ax.tick_params(colors='white')
                st.pyplot(fig)
            
            st.subheader("Key Frame Analysis")
            key_idx = res.get("phases",{}).get("end_of_pull_frame", len(raw_frames)//2 if raw_frames else 0)
            if key_idx < len(raw_frames):
                key_frame = raw_frames[key_idx]
                results = model.predict(key_frame, verbose=False)
                if len(all_kps) > key_idx and all_kps[key_idx] is not None:
                    results[0].keypoints = YOLO(model.ckpt_path).predictor.postprocess_pose([torch.from_numpy(all_kps[key_idx]).unsqueeze(0)], key_frame, key_frame)[0].keypoints
                annotated = results[0].plot()
                annotated = cv2.putText(annotated, res['verdict'], (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0) if res['verdict']=="Good Lift" else (0,0,255), 3)
                
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, caption=f"Key Frame at End of Pull (Frame {key_idx})")
                
                # Provide download for the image
                is_success, buffer = cv2.imencode(".jpg", annotated)
                io_buf = BytesIO(buffer)
                st.download_button(
                    label="Download Key Frame Image",
                    data=io_buf,
                    file_name="lift_analysis_keyframe.jpg",
                    mime="image/jpeg"
                )

    except Exception as e: st.error(f"Error: {e}")
    finally:
        if cap: cap.release()
        try: os.remove(video_path)
        except: pass
