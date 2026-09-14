import os
import sys
import cv2
import pickle
import numpy as np
import pandas as pd
import sqlite3
import time
import io
import urllib.parse
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, Response, jsonify, g, send_file, send_from_directory
)
import mediapipe as mp
try:
    import mediapipe.solutions.holistic as mp_holistic
    import mediapipe.solutions.pose as mp_pose
    import mediapipe.solutions.drawing_utils as mp_drawing
    import mediapipe.solutions.drawing_styles as mp_drawing_styles
except Exception:
    try:
        from mediapipe.python.solutions import holistic as mp_holistic
        from mediapipe.python.solutions import pose as mp_pose
        from mediapipe.python.solutions import drawing_utils as mp_drawing
        from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
    except Exception as e:
        mp_holistic = getattr(mp, 'solutions', None) and getattr(mp.solutions, 'holistic', None)
        mp_pose = getattr(mp, 'solutions', None) and getattr(mp.solutions, 'pose', None)
        mp_drawing = getattr(mp, 'solutions', None) and getattr(mp.solutions, 'drawing_utils', None)
        mp_drawing_styles = getattr(mp, 'solutions', None) and getattr(mp.solutions, 'drawing_styles', None)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pose_engine import BiomechanicalPoseEngine

app = Flask(__name__)
app.secret_key = 'yoga_studio_ai_physiotherapy_secret_key_2026'

# Path configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, 'evaluation.db')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'body_language.pkl')

# Global Pose Engine instance
pose_engine = BiomechanicalPoseEngine()

# Model loader helper
classifier_model = None
def load_ml_model():
    global classifier_model
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                classifier_model = pickle.load(f)
            print("[+] Loaded exercise classification model successfully.")
        except Exception as e:
            print(f"[!] Could not load ML model: {e}")
    else:
        print("[!] Warning: body_language.pkl model file not found yet.")

load_ml_model()

# Database Helper Functions
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fullname TEXT NOT NULL,
                username TEXT UNIQUE NOT NULL,
                email TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                exercise TEXT NOT NULL,
                form_score REAL NOT NULL,
                accuracy REAL NOT NULL,
                repetitions INTEGER NOT NULL,
                duration INTEGER NOT NULL,
                feedback TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        cursor.execute('SELECT COUNT(*) FROM users')
        if cursor.fetchone()[0] == 0:
            demo_hash = generate_password_hash('Demo123!')
            cursor.execute(
                'INSERT INTO users (fullname, username, email, password_hash) VALUES (?, ?, ?, ?)',
                ('Alex Morgan', 'demo', 'alex@physio.ai', demo_hash)
            )
            cursor.execute('''
                INSERT INTO sessions (user_id, exercise, form_score, accuracy, repetitions, duration, feedback)
                VALUES (1, 'Shoulder Rotation', 92.5, 96.0, 15, 180, '✓ Good shoulder alignment.')
            ''')
            conn.commit()
            print("[+] Initialized SQLite DB schema & seeded demo data.")

init_db()

# Global variables for camera stream state
current_exercise = "Shoulder Rotation"
latest_telemetry = {
    'form_score': 92.0,
    'confidence': 96.0,
    'rep_count': 0,
    'primary_feedback': 'Initializing AI Pose Engine...',
    'secondary_feedback': [],
    'voice_text': None,
    'metrics': {},
    'body_detected': False
}

def generate_camera_frames():
    global latest_telemetry, current_exercise, classifier_model
    
    if classifier_model is None:
        load_ml_model()

    # Attempt hardware camera #0 safely
    cap = None
    using_video_file = False
    try:
        temp_cap = cv2.VideoCapture(0)
        if temp_cap and temp_cap.isOpened():
            ret, test_frame = temp_cap.read()
            if ret and test_frame is not None:
                cap = temp_cap
            else:
                temp_cap.release()
    except Exception as e:
        print(f"[!] Hardware camera unavailable: {e}")
        cap = None

    # If hardware camera unavailable (Hosted / Cloud environment), use video fallback
    if cap is None:
        print("[!] Hardware camera 0 unavailable (Cloud / Hosted mode). Searching for video fallback...")
        video_dir = os.path.join(PROJECT_ROOT, 'videos')
        demo_videos = ['shoulder.mp4', 'cat_cow.mp4', 'spinal.mp4']
        chosen_video = None
        if os.path.exists(video_dir):
            for v in demo_videos:
                v_path = os.path.join(video_dir, v)
                if os.path.exists(v_path):
                    chosen_video = v_path
                    break
        
        if chosen_video:
            try:
                print(f"[+] Using hosted video fallback: {chosen_video}")
                cap = cv2.VideoCapture(chosen_video)
                using_video_file = True
            except Exception as e:
                print(f"[!] Could not open video file fallback: {e}")
                cap = None
        else:
            print("[!] No video fallback found. Operating synthetic frame mode...")
            cap = None

    holistic = get_holistic_detector()
    pose_fallback = get_pose_detector()
    while True:
            try:
                frame = None
                if cap and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        if using_video_file:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = cap.read()
                        if not ret:
                            frame = None

                if frame is None:
                    # Generate synthetic canvas if no video source is available
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "LIVE AI ASSESSMENT (HOSTED DEMO)", (80, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 242, 254), 2)
                    cv2.putText(frame, "Hosted Cloud Server - Camera Virtualized", (100, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                    cv2.putText(frame, f"Active Exercise: {current_exercise}", (140, 280),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 118), 1)
                    time.sleep(0.06)
                elif not using_video_file:
                    # Mirror camera view for webcam
                    frame = cv2.flip(frame, 1)

                h, w, c = frame.shape

                # RGB for MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False

                results = None
                if holistic:
                    results = holistic.process(image_rgb)
                
                pose_landmarks = results.pose_landmarks if results else None
                face_landmarks = results.face_landmarks if results else None
                left_hand_landmarks = results.left_hand_landmarks if results else None
                right_hand_landmarks = results.right_hand_landmarks if results else None

                if not pose_landmarks and pose_fallback:
                    results_pose = pose_fallback.process(image_rgb)
                    if results_pose and results_pose.pose_landmarks:
                        pose_landmarks = results_pose.pose_landmarks

                image_rgb.flags.writeable = True
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                raw_confidence = 0.96
                if pose_landmarks and face_landmarks and classifier_model is not None:
                    try:
                        pose_list = pose_landmarks.landmark
                        pose_row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_list]).flatten())
                        face_list = face_landmarks.landmark
                        face_row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in face_list]).flatten())

                        row = pose_row + face_row
                        X_sample = pd.DataFrame([row])
                        
                        body_language_prob = classifier_model.predict_proba(X_sample)[0]
                        raw_confidence = float(np.max(body_language_prob))
                    except Exception as e:
                        raw_confidence = 0.95

                # Evaluate form using biomechanical engine
                telemetry = pose_engine.evaluate_pose(
                    pose_landmarks,
                    current_exercise,
                    raw_confidence
                )
                latest_telemetry = telemetry

                # RENDER ULTRA-DETAILED SKELETON & MESH
                if face_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        face_landmarks,
                        mp_holistic.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )

                if left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        left_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 242, 254), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 230, 118), thickness=2)
                    )

                if right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 242, 254), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 230, 118), thickness=2)
                    )

                if pose_landmarks:
                    pose_conn_spec = mp_drawing.DrawingSpec(color=(254, 242, 0), thickness=3, circle_radius=3)
                    pose_lm_spec = mp_drawing.DrawingSpec(color=(255, 180, 0), thickness=3, circle_radius=4)
                    
                    mp_drawing.draw_landmarks(
                        image,
                        pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=pose_lm_spec,
                        connection_drawing_spec=pose_conn_spec
                    )

                    lm = pose_landmarks.landmark
                    
                    # 5. DUAL VISUAL CAMERA GUIDE: RED CROSS FOR WRONG VS GREEN TARGET FOR CORRECT
                    error_joints = telemetry.get('error_joints', [])
                    target_guides = telemetry.get('target_guides', [])

                    # Render RED CROSS on error joints
                    for ej in error_joints:
                        if isinstance(ej, int) and ej < len(lm):
                            cx, cy = int(lm[ej].x * w), int(lm[ej].y * h)
                            cv2.circle(image, (cx, cy), 22, (82, 82, 255), 3)
                            cv2.line(image, (cx - 10, cy - 10), (cx + 10, cy + 10), (82, 82, 255), 3)
                            cv2.line(image, (cx - 10, cy + 10), (cx + 10, cy - 10), (82, 82, 255), 3)
                            cv2.putText(image, "WRONG POSTURE", (cx - 40, cy - 28),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (82, 82, 255), 2)

                    # Render GREEN ARROW & TARGET GUIDELINE showing desired correction
                    for guide in target_guides:
                        from_idx = guide.get('idx')
                        if from_idx < len(lm):
                            cx, cy = int(lm[from_idx].x * w), int(lm[from_idx].y * h)
                            tx, ty = int(guide['target_x'] * w), int(guide['target_y'] * h)
                            cv2.arrowedLine(image, (cx, cy), (tx, ty), (0, 230, 118), 3, tipLength=0.3)
                            cv2.circle(image, (tx, ty), 12, (0, 230, 118), -1)
                            cv2.putText(image, "CORRECT ACTION", (tx + 12, ty + 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 118), 2)

                # Draw AI Telemetry overlay on video feed
                cv2.rectangle(image, (0, 0), (w, 50), (15, 20, 30), -1)
                cv2.putText(image, f"AI PHYSIO: {current_exercise.upper()}", (15, 33),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (254, 242, 0), 2)
                
                score_text = f"SCORE: {int(telemetry['form_score'])}% | REPS: {telemetry['rep_count']}"
                cv2.putText(image, score_text, (w - 280, 33),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 118), 2)

                fb_text = telemetry['primary_feedback']
                box_color = (15, 20, 30) if "✓" in fb_text else (30, 20, 80)
                cv2.rectangle(image, (0, h - 45), (w, h), box_color, -1)
                text_color = (0, 230, 118) if "✓" in fb_text else (82, 183, 255)
                cv2.putText(image, fb_text, (15, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2)

                ret, buffer = cv2.imencode('.jpg', image)
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as loop_err:
                print(f"[!] Frame stream loop exception: {loop_err}")
                err_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(err_frame, "AI ASSESSMENT STREAM (RECOVERY MODE)", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 242, 254), 2)
                ret, buffer = cv2.imencode('.jpg', err_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.1)

    if cap:
        cap.release()

# Helper to ensure demo user session for guest direct links
def ensure_authenticated_user():
    if 'user_id' not in session:
        session['user_id'] = 1
        session['username'] = 'demo'
        session['fullname'] = 'Alex Morgan'

# --- Flask Routes ---

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username').strip()
        password = request.form.get('password')

        db = get_db()
        user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()

        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['fullname'] = user['fullname']
            flash(f"Welcome back, {user['fullname']}!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid username or password.", "error")

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        fullname = request.form.get('fullname').strip()
        username = request.form.get('username').strip()
        email = request.form.get('email').strip()
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template('registration.html')

        db = get_db()
        existing_user = db.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
        if existing_user:
            flash("Username already taken. Please choose another.", "warning")
            return render_template('registration.html')

        password_hash = generate_password_hash(password)
        db.execute(
            'INSERT INTO users (fullname, username, email, password_hash) VALUES (?, ?, ?, ?)',
            (fullname, username, email, password_hash)
        )
        db.commit()

        flash("Account created successfully. Welcome to Yoga Studio!", "success")
        return redirect(url_for('login'))

    return render_template('registration.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out safely.", "info")
    return redirect(url_for('landing'))

@app.route('/dashboard')
def dashboard():
    ensure_authenticated_user()
    db = get_db()
    user_id = session['user_id']
    
    recent_sessions = db.execute('''
        SELECT * FROM sessions WHERE user_id = ? ORDER BY created_at DESC LIMIT 5
    ''', (user_id,)).fetchall()

    stats = db.execute('''
        SELECT 
            COUNT(*) as total_sessions,
            ROUND(AVG(form_score), 1) as avg_score,
            SUM(repetitions) as total_reps
        FROM sessions WHERE user_id = ?
    ''', (user_id,)).fetchone()

    return render_template('dashboard.html', sessions=recent_sessions, stats=stats)

@app.route('/assessment')
def assessment():
    ensure_authenticated_user()

    global current_exercise
    exercise = request.args.get('exercise', 'Shoulder Rotation')
    current_exercise = exercise
    try:
        pose_engine.reset_counter(current_exercise)
    except Exception as e:
        print(f"[!] Pose counter reset warning: {e}")

    exercises_16 = pose_engine.exercises_list
    return render_template('assessment.html', exercise=current_exercise, exercises_list=exercises_16)

def make_json_serializable(obj):
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [make_json_serializable(item) for item in obj]
    return obj

# Global MediaPipe detector instances
global_holistic = None
global_pose = None

def get_holistic_detector():
    global global_holistic
    if global_holistic is None and mp_holistic is not None:
        try:
            global_holistic = mp_holistic.Holistic(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
            print("[+] Initialized persistent MediaPipe Holistic detector (static_image_mode=True).")
        except Exception as e:
            print(f"[!] Error initializing MediaPipe Holistic: {e}")
    return global_holistic

def get_pose_detector():
    global global_pose
    if global_pose is None and mp_pose is not None:
        try:
            global_pose = mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
            print("[+] Initialized persistent MediaPipe Pose fallback detector.")
        except Exception as e:
            print(f"[!] Error initializing MediaPipe Pose fallback: {e}")
    return global_pose

import base64

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global latest_telemetry, current_exercise, classifier_model

    if classifier_model is None:
        load_ml_model()

    data = request.get_json() or {}
    image_data = data.get('image', '')
    if not image_data:
        clean_telemetry = make_json_serializable(latest_telemetry)
        return jsonify({'status': 'success', 'telemetry': clean_telemetry}), 200

    try:
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            clean_telemetry = make_json_serializable(latest_telemetry)
            return jsonify({'status': 'success', 'telemetry': clean_telemetry}), 200

        h, w, c = frame.shape

        holistic = get_holistic_detector()
        pose_fallback = get_pose_detector()

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        results = None
        if holistic:
            results = holistic.process(image_rgb)
        
        pose_landmarks = results.pose_landmarks if results else None
        face_landmarks = results.face_landmarks if results else None
        left_hand_landmarks = results.left_hand_landmarks if results else None
        right_hand_landmarks = results.right_hand_landmarks if results else None

        # Pose fallback detector if holistic misses pose landmarks
        if not pose_landmarks and pose_fallback:
            results_pose = pose_fallback.process(image_rgb)
            if results_pose and results_pose.pose_landmarks:
                pose_landmarks = results_pose.pose_landmarks

        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        raw_confidence = 0.96
        if pose_landmarks and face_landmarks and classifier_model is not None:
            try:
                pose_list = pose_landmarks.landmark
                pose_row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_list]).flatten())
                face_list = face_landmarks.landmark
                face_row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in face_list]).flatten())

                row = pose_row + face_row
                X_sample = pd.DataFrame([row])
                
                body_language_prob = classifier_model.predict_proba(X_sample)[0]
                raw_confidence = float(np.max(body_language_prob))
            except Exception as e:
                raw_confidence = 0.95

        telemetry = pose_engine.evaluate_pose(
            pose_landmarks,
            current_exercise,
            raw_confidence
        )
        latest_telemetry = telemetry

        # Draw Mesh and Skeleton
        if face_landmarks:
            mp_drawing.draw_landmarks(
                image,
                face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

        if left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 242, 254), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 230, 118), thickness=2)
            )

        if right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 242, 254), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 230, 118), thickness=2)
            )

        if pose_landmarks:
            pose_conn_spec = mp_drawing.DrawingSpec(color=(254, 242, 0), thickness=3, circle_radius=3)
            pose_lm_spec = mp_drawing.DrawingSpec(color=(255, 180, 0), thickness=3, circle_radius=4)
            
            mp_drawing.draw_landmarks(
                image,
                pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=pose_lm_spec,
                connection_drawing_spec=pose_conn_spec
            )

            lm = pose_landmarks.landmark
            error_joints = telemetry.get('error_joints', [])
            target_guides = telemetry.get('target_guides', [])

            for ej in error_joints:
                if isinstance(ej, int) and ej < len(lm):
                    cx, cy = int(lm[ej].x * w), int(lm[ej].y * h)
                    cv2.circle(image, (cx, cy), 22, (82, 82, 255), 3)
                    cv2.line(image, (cx - 10, cy - 10), (cx + 10, cy + 10), (82, 82, 255), 3)
                    cv2.line(image, (cx - 10, cy + 10), (cx + 10, cy - 10), (82, 82, 255), 3)
                    cv2.putText(image, "WRONG POSTURE", (cx - 40, cy - 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (82, 82, 255), 2)

            for guide in target_guides:
                from_idx = guide.get('idx')
                if from_idx < len(lm):
                    cx, cy = int(lm[from_idx].x * w), int(lm[from_idx].y * h)
                    tx, ty = int(guide['target_x'] * w), int(guide['target_y'] * h)
                    cv2.arrowedLine(image, (cx, cy), (tx, ty), (0, 230, 118), 3, tipLength=0.3)
                    cv2.circle(image, (tx, ty), 12, (0, 230, 118), -1)
                    cv2.putText(image, "CORRECT ACTION", (tx + 12, ty + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 118), 2)

        cv2.rectangle(image, (0, 0), (w, 50), (15, 20, 30), -1)
        cv2.putText(image, f"AI PHYSIO: {current_exercise.upper()}", (15, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (254, 242, 0), 2)
        
        score_text = f"SCORE: {int(telemetry['form_score'])}% | REPS: {telemetry['rep_count']}"
        cv2.putText(image, score_text, (w - 280, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 118), 2)

        fb_text = telemetry['primary_feedback']
        box_color = (15, 20, 30) if "✓" in fb_text else (30, 20, 80)
        cv2.rectangle(image, (0, h - 45), (w, h), box_color, -1)
        text_color = (0, 230, 118) if "✓" in fb_text else (82, 183, 255)
        cv2.putText(image, fb_text, (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2)

        ret, buffer = cv2.imencode('.jpg', image)
        if not ret:
            clean_telemetry = make_json_serializable(latest_telemetry)
            return jsonify({'status': 'success', 'telemetry': clean_telemetry}), 200

        processed_base64 = base64.b64encode(buffer).decode('utf-8')
        clean_telemetry = make_json_serializable(telemetry)

        res = jsonify({
            'status': 'success',
            'image': f'data:image/jpeg;base64,{processed_base64}',
            'telemetry': clean_telemetry
        })
        res.headers.add('Access-Control-Allow-Origin', '*')
        return res

    except Exception as e:
        print(f"[!] Error processing client frame: {e}")
        clean_telemetry = make_json_serializable(latest_telemetry)
        res = jsonify({
            'status': 'success',
            'image': data.get('image', ''),
            'telemetry': clean_telemetry
        })
        res.headers.add('Access-Control-Allow-Origin', '*')
        return res, 200

@app.route('/video_feed')
def video_feed():
    return Response(generate_camera_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/telemetry')
def telemetry():
    global latest_telemetry
    try:
        clean_telemetry = make_json_serializable(latest_telemetry)
        res = jsonify(clean_telemetry)
        res.headers.add('Access-Control-Allow-Origin', '*')
        return res
    except Exception as e:
        print(f"[!] Error serializing telemetry: {e}")
        fallback = {
            'form_score': 0,
            'confidence': 0,
            'rep_count': 0,
            'primary_feedback': "Initializing AI Pose Engine...",
            'secondary_feedback': [],
            'error_joints': [],
            'target_guides': [],
            'voice_text': None,
            'metrics': {},
            'body_detected': False
        }
        res = jsonify(fallback)
        res.headers.add('Access-Control-Allow-Origin', '*')
        return res

@app.route('/tts_audio')
def tts_audio():
    text = request.args.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang='en', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return send_file(fp, mimetype='audio/mp3')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/complete_session', methods=['POST'])
def complete_session():
    user_id = session.get('user_id', 1)

    data = request.get_json() or {}
    exercise_name = data.get('exercise', current_exercise)
    form_score = data.get('form_score', latest_telemetry.get('form_score', 92.0))
    accuracy = data.get('accuracy', latest_telemetry.get('confidence', 96.0))
    repetitions = data.get('repetitions', latest_telemetry.get('rep_count', 0))
    duration = data.get('duration', 120)
    feedback = data.get('feedback', latest_telemetry.get('primary_feedback', 'Completed exercise session.'))

    db = get_db()
    cursor = db.cursor()
    cursor.execute('''
        INSERT INTO sessions (user_id, exercise, form_score, accuracy, repetitions, duration, feedback)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, exercise_name, form_score, accuracy, repetitions, duration, feedback))
    db.commit()

    session_id = cursor.lastrowid
    return jsonify({'status': 'success', 'redirect_url': url_for('session_report', session_id=session_id)})

@app.route('/session_report/<int:session_id>')
def session_report(session_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    db = get_db()
    session_data = db.execute('SELECT * FROM sessions WHERE id = ? AND user_id = ?', 
                              (session_id, session['user_id'])).fetchone()

    if not session_data:
        flash("Session report not found.", "error")
        return redirect(url_for('dashboard'))

    return render_template('session_report.html', report=session_data)

@app.route('/exercises')
def exercises():
    exercises_16 = pose_engine.exercises_list
    return render_template('exercises.html', exercises_list=exercises_16)

@app.route('/exercise/<exercise_name>')
def exercise_detail(exercise_name):
    return render_template('exercise_detail.html', exercise_name=exercise_name)

@app.route('/yoga')
def yoga():
    return render_template('yoga.html')

@app.route('/paper')
def view_paper():
    paper_path = os.path.join(PROJECT_ROOT, 'IEEE_Research_Paper_AI_Physiotherapy.html')
    if os.path.exists(paper_path):
        return send_file(paper_path)
    return "IEEE Research Paper file not found", 404

@app.route('/download_paper')
def download_paper():
    paper_path = os.path.join(PROJECT_ROOT, 'IEEE_Research_Paper_AI_Physiotherapy.html')
    if os.path.exists(paper_path):
        return send_file(paper_path, as_attachment=True, download_name='IEEE_Research_Paper_AI_Physiotherapy.html')
    return "IEEE Research Paper file not found", 404

@app.route('/beacon', methods=['GET', 'POST'])
def beacon():
    return ('', 204)

@app.route('/rehabilitation')
def rehabilitation():
    return render_template('rehabilitation.html')

@app.route('/progress')
def progress():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    db = get_db()
    user_id = session['user_id']
    sessions_history = db.execute('''
        SELECT * FROM sessions WHERE user_id = ? ORDER BY created_at ASC
    ''', (user_id,)).fetchall()

    return render_template('progress.html', history=sessions_history)

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    db = get_db()
    user_id = session['user_id']
    user = db.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    
    stats = db.execute('''
        SELECT 
            COUNT(*) as total_sessions,
            ROUND(AVG(form_score), 1) as avg_score,
            SUM(repetitions) as total_reps,
            SUM(duration) as total_duration
        FROM sessions WHERE user_id = ?
    ''', (user_id,)).fetchone()

    return render_template('profile.html', user=user, stats=stats)

@app.route('/videos/<path:filename>')
def serve_video(filename):
    filename = urllib.parse.unquote(filename)
    video_dir = os.path.join(PROJECT_ROOT, 'videos')
    return send_from_directory(video_dir, filename)

@app.route('/media/<path:filename>')
def serve_media(filename):
    filename = urllib.parse.unquote(filename)
    new_img_dir = os.path.join(PROJECT_ROOT, 'NEW IMAGES')
    if os.path.exists(os.path.join(new_img_dir, filename)):
        return send_from_directory(new_img_dir, filename)
    img_dir = os.path.join(PROJECT_ROOT, 'Images')
    return send_from_directory(img_dir, filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print("=" * 60)
    print(f" AI Physiotherapy Platform (Yoga Studio) - Server Starting...")
    print(f" Access Web Application at http://127.0.0.1:{port}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=False)
