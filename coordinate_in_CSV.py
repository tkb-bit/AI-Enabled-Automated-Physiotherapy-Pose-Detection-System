import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import csv

def main():
    print("=" * 60)
    print(" AI Physiotherapy - Landmark Data Collector")
    print("=" * 60)
    
    exercise_class = input("Enter the exercise class label (e.g. 'Shoulder Rotation', 'Wrist Extension', 'Spinal Twist', 'Arm Rotation'): ").strip()
    if not exercise_class:
        exercise_class = "Shoulder Rotation"
        
    os.makedirs("dataset", exist_ok=True)
    csv_file = "dataset/coords1.csv"
    
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    # Determine num landmarks for header creation if file doesn't exist
    # 33 pose landmarks + 468 face landmarks = 501 landmarks (each x, y, z, visibility for pose, x, y, z for face)
    num_coords = 33 + 468
    
    if not os.path.exists(csv_file):
        landmarks = ['class']
        for val in range(1, num_coords + 1):
            landmarks.extend([f'x{val}', f'y{val}', f'z{val}', f'v{val}'])
        with open(csv_file, mode='w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(landmarks)
        print(f"[+] Created CSV file: {csv_file}")

    cap = cv2.VideoCapture(0)
    print(f"[+] Capturing dataset for class '{exercise_class}'. Press 'q' to quit...")
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[-] Failed to read webcam frame.")
                break
                
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw pose & face
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.face_landmarks:
                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)

            try:
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                row = pose_row + face_row
                row.insert(0, exercise_class)

                with open(csv_file, mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row)

                cv2.putText(image, f"Class: {exercise_class} (Recording...)", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            except Exception as e:
                cv2.putText(image, "Body/Face not detected!", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Landmark Data Collector", image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[+] Data collection finished.")

if __name__ == "__main__":
    main()
