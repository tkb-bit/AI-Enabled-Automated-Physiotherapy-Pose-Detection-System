# Yoga Studio — AI Physiotherapy & Automated Exercise Assessment Platform

> **Final-Year Engineering Project**: Real-time AI movement analysis, computer vision pose tracking, biomechanical form evaluation, error joint highlighting, spoken voice feedback, and progress tracking.

---

## 🌟 Key Features

1. **Real-Time Camera AI Assessment Room**:
   - MediaPipe Holistic pose detection rendering 33 full-body skeleton landmarks.
   - Pre-trained ML exercise classifier (`body_language.pkl`).
   - Biomechanical joint-angle & range-of-motion evaluation.
   - **Visual Joint Error Highlighting**: Pulsating red warning halos drawn directly on faulty joint positions (e.g., flared elbows, leaning spine).
   - Spoken Text-to-Speech (TTS) voice feedback with duplicate throttling.
   - Automated repetition counter & live 0–100% Form Score gauge.

2. **Personalized User Dashboard & Authentication**:
   - SQLite (`evaluation.db`) user accounts with Werkzeug password hashing.
   - Quick session launcher, progress summaries, and recent activity logs.

3. **Session Analytics & Reports**:
   - Post-session performance breakdown report.
   - Accomplishment checklists, needs-improvement alerts, and personalized AI exercise recommendations.

4. **Physiotherapy Exercise Library & Guided Yoga**:
   - Searchable exercise database (Shoulder Rotation, Wrist Extension, Spinal Twist, Arm Rotation).
   - Streaming video interface for guided yoga flows (Cat-Cow, Spinal Flexion).

5. **Rehabilitation & Safety Guide**:
   - Accordion guides for shoulder pain, wrist strain, spine care, and medical safety position disclaimers.

---

## 🚀 Technology Stack

- **Backend**: Python 3.11, Flask, SQLite3, OpenCV, MediaPipe, NumPy, Pandas, Scikit-learn, gTTS.
- **Frontend**: HTML5, Vanilla CSS3 (Dark Medical UX system with glassmorphism), Vanilla JavaScript.
- **Machine Learning**: Random Forest / Logistic Regression classifier trained on MediaPipe Holistic feature vectors (`dataset/coords1.csv`).

---

## 📁 Directory Structure

```text
ai yoga/
│
├── yoga_app/
│   ├── app.py                     # Flask web server & streaming API
│   ├── pose_engine.py             # Biomechanical form & error joint evaluator
│   │
│   ├── templates/                 # Jinja2 HTML templates
│   │   ├── base.html              # Master layout with dark theme & toasts
│   │   ├── landing.html           # Hero landing page
│   │   ├── login.html             # User login
│   │   ├── registration.html      # Account creation
│   │   ├── dashboard.html         # User dashboard
│   │   ├── assessment.html        # Live camera AI HUD room
│   │   ├── session_report.html    # Post-session report
│   │   ├── exercises.html         # Exercise library
│   │   ├── exercise_detail.html   # Detailed exercise guide
│   │   ├── yoga.html              # Guided yoga video hub
│   │   ├── rehabilitation.html    # Rehab accordion guides
│   │   ├── progress.html          # Analytics & progress chart
│   │   └── profile.html           # User profile
│   │
│   └── static/
│       ├── styles.css             # Master dark healthcare design system
│       └── app.js                 # Live HUD telemetry & voice controller
│
├── dataset/
│   └── coords1.csv                # Landmark training dataset
│
├── body_language.pkl              # Pre-trained ML exercise classifier
├── training.py                    # ML model benchmark & training pipeline
├── coordinate_in_CSV.py           # Landmark CSV data collector
├── human_skeleten.py              # Standalone OpenCV skeleton visualizer
├── evaluation.db                  # SQLite database
├── requirements.txt               # Dependencies
└── README.md
```

---

## 🛠️ How to Run the Project

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train / Generate ML Model** *(Optional - pre-generated model included)*:
   ```bash
   python training.py
   ```

3. **Start the Web Application**:
   ```bash
   python yoga_app/app.py
   ```

4. **Access Web Application**:
   Open browser at: `http://127.0.0.1:5000`

---

## 🎬 Live Demonstration Flow

1. Open `http://127.0.0.1:5000` -> Explore **Landing Page**.
2. Click **Get Started** -> Log in with `demo` / `Demo123!`.
3. Click **Start Session Now** on Dashboard -> Select **Shoulder Rotation**.
4. Allow webcam access -> Observe live **MediaPipe Skeleton Overlay**.
5. Perform movement correctly -> Observe **Form Score 90%+** and **Repetition Counter**.
6. Intentionally flare elbow -> Observe **Red Glowing Joint Circle** over elbow and spoken voice feedback: *"Keep your elbow closer to your body."*
7. Click **End Session & Save Report** -> View comprehensive **Session Report**.
8. Inspect **Progress Analytics** & **Rehab Guide**.

---

## 👥 Contributors

- **Kartik Suryavanshi** ([@KartikSuryavanshi](https://github.com/KartikSuryavanshi/)) — Core Contributor
- **Tushilesh Borse** ([@tkb-bit](https://github.com/tkb-bit/)) — Core Contributor

