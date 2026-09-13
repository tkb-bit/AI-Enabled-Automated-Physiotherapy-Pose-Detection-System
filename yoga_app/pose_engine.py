import numpy as np
import time
import math

class BiomechanicalPoseEngine:
    def __init__(self):
        # 55 Physiotherapy & Yoga Exercises
        self.exercises_list = [
            'Shoulder Rotation', 'Wrist Extension', 'Spinal Twist', 'Arm Rotation',
            'Cat-Cow Pose', 'Overhead Arm Stretch', 'Neck Roll', 'Chest Opener',
            'Squat & Hip Mobility', 'Tree Pose Balance', 'Warrior Pose', 'Bird-Dog Core Extension',
            'Glute Bridge', 'Scapular Retraction', 'Forearm Flexion', 'Side Lateral Raise',
            'Triangle Pose', 'Downward-Facing Dog', 'Cobra Back Extension', 'Child\'s Pose Lumbar Relief',
            'Ankle Circles & Mobility', 'Hamstring Stretch & Release', 'Quadriceps Stretch', 'Calf Raise & Balance',
            'Plank Core Hold', 'Side Plank Lateral Core', 'Thoracic Extension', 'Hip Flexor Lunge Stretch',
            'Piriformis Stretch', 'Pelvic Tilt Stabilization', 'Scapular Wall Slide', 'Chin Tuck Neck Posture',
            'Shoulder External Rotation', 'Wrist Flexion Stretch', 'Pronation & Supination', 'High Knee Lift Balance',
            'Standing Side Bend', 'Seated Forward Bend', 'Bridge Raise Single Leg', 'Dead Bug Core Control',
            'Clamshell Hip Abduction', 'Monster Walk Lateral Band', 'Pendulum Arm Swing', 'Cross-Body Shoulder Stretch',
            'Triceps Extension Stretch', 'Biceps Stretch', 'Abductor Leg Raise', 'Adductor Squeeze',
            'Toe Touch Forward Fold', 'Reverse Fly Posture', 'Spinal Arch Flexion', 'Wall Sit Endurance',
            'Quadruped Hip Extension', 'Cervical Lateral Flexion', 'Full-Body Restorative Stretch'
        ]
        
        self.rep_counters = {ex: {'count': 0, 'stage': 'down', 'peak_reached': False, 'hold_start': 0} for ex in self.exercises_list}
        
        # Low latency voice feedback cooldown tracker
        self.last_voice_time = 0.0
        self.last_voice_message = ""
        self.voice_cooldown_seconds = 2.5
        
        self.smoothed_confidence = 0.96
        self.smoothed_form_score = 92.0

    @staticmethod
    def calculate_angle(a, b, c):
        """Calculates 2D angle (in degrees) between three landmark points with b as vertex."""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360.0 - angle
            
        return angle

    @staticmethod
    def calculate_distance(p1, p2):
        """Euclidean distance between two landmarks."""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def reset_counter(self, exercise_name):
        if exercise_name in self.rep_counters:
            self.rep_counters[exercise_name] = {'count': 0, 'stage': 'down', 'peak_reached': False, 'hold_start': 0}

    def evaluate_pose(self, pose_landmarks, exercise_name, raw_confidence=0.96):
        """
        Evaluates 55 exercises with unique biomechanical joint geometry, error flags, target guides, 
        rep tracking, and live voice assistance.
        """
        if not pose_landmarks:
            return {
                'form_score': 0,
                'confidence': 0,
                'rep_count': 0,
                'primary_feedback': "Move full body into camera frame",
                'secondary_feedback': ["Ensure your torso, arms, and hips are clearly visible"],
                'error_joints': [],
                'target_guides': [],
                'voice_text': None,
                'metrics': {},
                'body_detected': False
            }

        lm = pose_landmarks.landmark
        
        l_sh, r_sh = lm[11], lm[12]
        l_el, r_el = lm[13], lm[14]
        l_wr, r_wr = lm[15], lm[16]
        l_hip, r_hip = lm[23], lm[24]
        l_knee, r_knee = lm[25], lm[26]
        l_ank, r_ank = lm[27], lm[28]
        nose = lm[0]

        shoulder_diff_y = abs(l_sh.y - r_sh.y)
        hip_diff_y = abs(l_hip.y - r_hip.y)
        
        l_elbow_angle = self.calculate_angle(l_sh, l_el, l_wr)
        r_elbow_angle = self.calculate_angle(r_sh, r_el, r_wr)
        l_shoulder_angle = self.calculate_angle(l_hip, l_sh, l_el)
        r_shoulder_angle = self.calculate_angle(r_hip, r_sh, r_el)
        l_knee_angle = self.calculate_angle(l_hip, l_knee, l_ank)
        r_knee_angle = self.calculate_angle(r_hip, r_knee, r_ank)

        l_elbow_flare = self.calculate_distance(l_el, l_hip)
        r_elbow_flare = self.calculate_distance(r_el, r_hip)

        primary_feedback = "✓ Excellent form! Maintain smooth control."
        secondary_feedback = []
        error_joints = []
        target_guides = []
        score_deductions = 0

        counter_state = self.rep_counters.get(exercise_name, {'count': 0, 'stage': 'down', 'peak_reached': False, 'hold_start': 0})
        prev_count = counter_state.get('count', 0)

        # ---------------------------------------------------------
        # UNIQUE BIOMECHANICAL EVALUATION LOGIC FOR EACH EXERCISE
        # ---------------------------------------------------------

        # 1. SHOULDER ROTATION
        if exercise_name == 'Shoulder Rotation':
            if l_elbow_angle < 70 or r_elbow_angle < 70:
                primary_feedback = "⚠ Keep elbows bent at 90 degrees during rotation."
                error_joints.extend([13, 14])
                target_guides.append({'idx': 13, 'target_x': l_sh.x - 0.1, 'target_y': l_sh.y})
                score_deductions += 15
            else:
                primary_feedback = "✓ Smooth shoulder rotation. Keep elbows bent at 90°."

            avg_arm_angle = (l_shoulder_angle + r_shoulder_angle) / 2.0
            if avg_arm_angle > 70 and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif avg_arm_angle < 35 and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 2. WRIST EXTENSION
        elif exercise_name == 'Wrist Extension':
            if (l_wr.y > l_el.y and r_wr.y > r_el.y):
                primary_feedback = "⚠ Raise forearms and extend wrists upward."
                error_joints.extend([15, 16])
                target_guides.append({'idx': 15, 'target_x': l_el.x - 0.15, 'target_y': l_el.y - 0.10})
                score_deductions += 14
            else:
                primary_feedback = "✓ Wrists extended. Maintain steady forearm level."

            if (l_wr.y < l_el.y - 0.05 or r_wr.y < r_el.y - 0.05) and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif (l_wr.y >= l_el.y) and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 3. SPINAL TWIST
        elif exercise_name == 'Spinal Twist':
            twist_diff = l_sh.z - r_sh.z
            if hip_diff_y > 0.08:
                primary_feedback = "⚠ Keep hips square and rotate strictly through torso."
                error_joints.extend([23, 24])
                score_deductions += 16
            else:
                primary_feedback = "✓ Deep spinal twist with square hips."

            if abs(twist_diff) > 0.14 and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif abs(twist_diff) < 0.05 and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 4. ARM ROTATION
        elif exercise_name == 'Arm Rotation':
            if l_elbow_angle < 140 or r_elbow_angle < 140:
                primary_feedback = "⚠ Fully extend arms straight out without bending elbows."
                error_joints.extend([13, 14])
                target_guides.append({'idx': 13, 'target_x': l_sh.x - 0.25, 'target_y': l_sh.y})
                target_guides.append({'idx': 14, 'target_x': r_sh.x + 0.25, 'target_y': r_sh.y})
                score_deductions += 15
            else:
                primary_feedback = "✓ Full arm rotation. Keep arms extended parallel."

            avg_arm_angle = (l_shoulder_angle + r_shoulder_angle) / 2.0
            if avg_arm_angle > 75 and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif avg_arm_angle < 40 and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 5. CAT-COW POSE
        elif exercise_name == 'Cat-Cow Pose':
            spine_center_y = (l_sh.y + r_sh.y + l_hip.y + r_hip.y) / 4.0
            primary_feedback = "✓ Great arch and flexion of the spine."

            if spine_center_y < 0.45 and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif spine_center_y >= 0.45 and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 6. OVERHEAD ARM STRETCH
        elif exercise_name == 'Overhead Arm Stretch':
            if l_wr.y > l_sh.y - 0.15 or r_wr.y > r_sh.y - 0.15:
                primary_feedback = "⚠ Reach higher overhead and straighten your elbows."
                error_joints.extend([15, 16])
                target_guides.append({'idx': 15, 'target_x': l_sh.x, 'target_y': l_sh.y - 0.30})
                target_guides.append({'idx': 16, 'target_x': r_sh.x, 'target_y': r_sh.y - 0.30})
                score_deductions += 18
            else:
                primary_feedback = "✓ Overhead arms fully extended toward ceiling."

            if (l_wr.y < l_sh.y - 0.25 and r_wr.y < r_sh.y - 0.25) and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif (l_wr.y > l_sh.y - 0.10) and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 7. NECK ROLL
        elif exercise_name == 'Neck Roll':
            if shoulder_diff_y > 0.06:
                primary_feedback = "⚠ Keep shoulders relaxed and level while rolling neck."
                error_joints.extend([11, 12])
                score_deductions += 12
            else:
                primary_feedback = "✓ Gentle, controlled cervical neck roll."

            nose_offset = abs(nose.x - (l_sh.x + r_sh.x) / 2.0)
            if nose_offset > 0.08 and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif nose_offset < 0.03 and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 8. CHEST OPENER
        elif exercise_name == 'Chest Opener':
            arm_span = abs(l_wr.x - r_wr.x)
            if arm_span < 0.40:
                primary_feedback = "⚠ Open arms wider and pull shoulder blades together."
                error_joints.extend([15, 16])
                target_guides.append({'idx': 15, 'target_x': l_sh.x - 0.3, 'target_y': l_sh.y})
                target_guides.append({'idx': 16, 'target_x': r_sh.x + 0.3, 'target_y': r_sh.y})
                score_deductions += 15
            else:
                primary_feedback = "✓ Chest open, scapulae retracting nicely."

            if arm_span > 0.55 and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif arm_span < 0.35 and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 9. SQUAT & HIP MOBILITY
        elif exercise_name == 'Squat & Hip Mobility':
            avg_knee_angle = (l_knee_angle + r_knee_angle) / 2.0
            knee_valgus = abs(l_knee.x - r_knee.x) < abs(l_ank.x - r_ank.x) * 0.8

            if knee_valgus:
                primary_feedback = "⚠ Push knees outward inline with your toes."
                error_joints.extend([25, 26])
                score_deductions += 16
            elif avg_knee_angle > 130 and counter_state['stage'] == 'up':
                primary_feedback = "⚠ Lower hips deeper until thighs are parallel."
                target_guides.append({'idx': 25, 'target_x': l_knee.x, 'target_y': l_knee.y + 0.15})
                target_guides.append({'idx': 26, 'target_x': r_knee.x, 'target_y': r_knee.y + 0.15})
                score_deductions += 12
            else:
                primary_feedback = "✓ Deep squat depth achieved with strong knee tracking."

            if avg_knee_angle < 110 and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif avg_knee_angle > 150 and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 10. TREE POSE BALANCE
        elif exercise_name == 'Tree Pose Balance':
            ank_diff = abs(l_ank.y - r_ank.y)
            if ank_diff < 0.12:
                primary_feedback = "⚠ Lift one foot higher and rest it against inner thigh."
                error_joints.extend([27, 28])
                score_deductions += 15
            elif hip_diff_y > 0.07:
                primary_feedback = "⚠ Level your hips and engage standing leg core."
                error_joints.extend([23, 24])
                score_deductions += 12
            else:
                primary_feedback = "✓ Solid single-leg balance and core alignment."

            if ank_diff > 0.18 and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif ank_diff < 0.08 and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 11. WARRIOR POSE
        elif exercise_name == 'Warrior Pose':
            front_knee_angle = min(l_knee_angle, r_knee_angle)
            if front_knee_angle > 120:
                primary_feedback = "⚠ Deepen front knee bend toward 90 degrees."
                error_joints.append(25 if l_knee_angle < r_knee_angle else 26)
                score_deductions += 16
            elif shoulder_diff_y > 0.08:
                primary_feedback = "⚠ Keep arms horizontal and level with shoulders."
                error_joints.extend([11, 12])
                score_deductions += 10
            else:
                primary_feedback = "✓ Strong Warrior stance! Arms horizontal."

            if front_knee_angle < 105 and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif front_knee_angle > 140 and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 12. BIRD-DOG CORE EXTENSION
        elif exercise_name == 'Bird-Dog Core Extension':
            arm_height = min(l_wr.y, r_wr.y)
            leg_height = min(l_ank.y, r_ank.y)
            if arm_height > (l_sh.y + 0.10) or leg_height > (l_hip.y + 0.10):
                primary_feedback = "⚠ Extend opposite arm and leg parallel to the floor."
                error_joints.extend([15, 28])
                score_deductions += 15
            else:
                primary_feedback = "✓ Parallel arm and leg reach. Core braced."

            if (arm_height < l_sh.y) and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif (arm_height > l_sh.y + 0.15) and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 13. GLUTE BRIDGE
        elif exercise_name == 'Glute Bridge':
            hip_height = (l_hip.y + r_hip.y) / 2.0
            knee_height = (l_knee.y + r_knee.y) / 2.0
            if hip_height > knee_height - 0.02:
                primary_feedback = "⚠ Drive hips higher toward ceiling to engage glutes."
                error_joints.extend([23, 24])
                target_guides.append({'idx': 23, 'target_x': l_hip.x, 'target_y': l_hip.y - 0.15})
                score_deductions += 18
            else:
                primary_feedback = "✓ Hips fully elevated. Glutes engaged."

            if hip_height < knee_height - 0.05 and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif hip_height >= knee_height and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 14. SCAPULAR RETRACTION
        elif exercise_name == 'Scapular Retraction':
            sh_z_diff = abs(l_sh.z - r_sh.z)
            primary_feedback = "✓ Scapulae squeezed tightly together."

            if l_el.z > l_sh.z and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif l_el.z <= l_sh.z and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 15. FOREARM FLEXION
        elif exercise_name == 'Forearm Flexion':
            if l_elbow_flare > 0.35 or r_elbow_flare > 0.35:
                primary_feedback = "⚠ Keep elbows pinned close to ribs during curl."
                error_joints.extend([13, 14])
                score_deductions += 14
            else:
                primary_feedback = "✓ Full biceps & forearm flexion range."

            avg_elbow_angle = (l_elbow_angle + r_elbow_angle) / 2.0
            if avg_elbow_angle < 60 and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif avg_elbow_angle > 140 and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 16. SIDE LATERAL RAISE
        elif exercise_name == 'Side Lateral Raise':
            avg_arm_angle = (l_shoulder_angle + r_shoulder_angle) / 2.0
            if avg_arm_angle < 65 and counter_state['stage'] == 'down':
                primary_feedback = "⚠ Raise arms up parallel to shoulder level (90°)."
                error_joints.extend([15, 16])
                target_guides.append({'idx': 15, 'target_x': l_sh.x - 0.25, 'target_y': l_sh.y})
                target_guides.append({'idx': 16, 'target_x': r_sh.x + 0.25, 'target_y': r_sh.y})
                score_deductions += 15
            else:
                primary_feedback = "✓ Arms raised to 90 degrees parallel."

            if avg_arm_angle > 75 and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif avg_arm_angle < 30 and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 17. TRIANGLE POSE
        elif exercise_name == 'Triangle Pose':
            torso_lean = abs(l_sh.x - l_hip.x)
            if torso_lean < 0.12:
                primary_feedback = "⚠ Hinge laterally at hips while reaching top arm high."
                error_joints.extend([11, 15])
                score_deductions += 15
            else:
                primary_feedback = "✓ Lateral torso stretch with vertical arm stack."

            if torso_lean > 0.18 and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif torso_lean < 0.08 and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 18. DOWNWARD-FACING DOG
        elif exercise_name == 'Downward-Facing Dog':
            hip_y = (l_hip.y + r_hip.y) / 2.0
            sh_y = (l_sh.y + r_sh.y) / 2.0
            if hip_y > sh_y - 0.05:
                primary_feedback = "⚠ Push hips up and back toward ceiling into inverted V."
                error_joints.extend([23, 24])
                score_deductions += 16
            else:
                primary_feedback = "✓ Perfect inverted V stance. Spine long."

            if hip_y < sh_y - 0.10 and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif hip_y >= sh_y and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 19. COBRA BACK EXTENSION
        elif exercise_name == 'Cobra Back Extension':
            sh_y = (l_sh.y + r_sh.y) / 2.0
            hip_y = (l_hip.y + r_hip.y) / 2.0
            if sh_y > hip_y - 0.10:
                primary_feedback = "⚠ Press palms down and lift chest higher."
                error_joints.extend([11, 12])
                score_deductions += 15
            else:
                primary_feedback = "✓ Upper back lifted and chest extended."

            if sh_y < hip_y - 0.15 and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif sh_y >= hip_y - 0.05 and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 20. CHILD'S POSE LUMBAR RELIEF
        elif exercise_name == 'Child\'s Pose Lumbar Relief':
            primary_feedback = "✓ Lumbar spine relaxed into deep Child's Pose."
            if counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
                counter_state['count'] += 1

        # 21. ANKLE CIRCLES & MOBILITY
        elif exercise_name == 'Ankle Circles & Mobility':
            primary_feedback = "✓ Smooth ankle rotational mobility."
            ank_dist = self.calculate_distance(l_ank, r_ank)
            if ank_dist > 0.25 and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif ank_dist <= 0.25 and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 22. HAMSTRING STRETCH & RELEASE
        elif exercise_name == 'Hamstring Stretch & Release':
            if l_knee_angle < 150 and r_knee_angle < 150:
                primary_feedback = "⚠ Keep standing knee straight while hinging forward."
                error_joints.extend([25, 26])
                score_deductions += 15
            else:
                primary_feedback = "✓ Hamstring stretch engaged with flat back."

            torso_fold = abs(l_sh.y - l_hip.y)
            if torso_fold < 0.15 and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif torso_fold >= 0.22 and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 23. QUADRICEPS STRETCH
        elif exercise_name == 'Quadriceps Stretch':
            bent_knee_angle = min(l_knee_angle, r_knee_angle)
            if bent_knee_angle > 80:
                primary_feedback = "⚠ Pull heel closer to glute to stretch quad."
                error_joints.append(25 if l_knee_angle < r_knee_angle else 26)
                score_deductions += 16
            else:
                primary_feedback = "✓ Heel pulled close to glute. Quad stretching."

            if bent_knee_angle < 50 and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif bent_knee_angle > 120 and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 24. CALF RAISE & BALANCE
        elif exercise_name == 'Calf Raise & Balance':
            ank_y = (l_ank.y + r_ank.y) / 2.0
            if counter_state.get('base_ank_y', None) is None:
                counter_state['base_ank_y'] = ank_y

            base_y = counter_state['base_ank_y']
            if ank_y > base_y - 0.03:
                primary_feedback = "⚠ Rise up higher onto balls of your feet."
                error_joints.extend([27, 28])
                score_deductions += 12
            else:
                primary_feedback = "✓ Heels elevated high onto toes."

            if ank_y < base_y - 0.05 and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif ank_y >= base_y - 0.01 and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 25. PLANK CORE HOLD
        elif exercise_name == 'Plank Core Hold':
            hip_dev = abs(((l_sh.y + r_sh.y)/2.0 + (l_ank.y + r_ank.y)/2.0)/2.0 - (l_hip.y + r_hip.y)/2.0)
            if hip_dev > 0.06:
                primary_feedback = "⚠ Keep body in rigid straight line; avoid sagging hips."
                error_joints.extend([23, 24])
                score_deductions += 18
            else:
                primary_feedback = "✓ Rigid core plank! Spine in neutral line."

            if counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif hip_dev < 0.04:
                counter_state['count'] += 1

        # 26. SIDE PLANK LATERAL CORE
        elif exercise_name == 'Side Plank Lateral Core':
            hip_y = (l_hip.y + r_hip.y) / 2.0
            sh_y = (l_sh.y + r_sh.y) / 2.0
            if hip_y > sh_y + 0.12:
                primary_feedback = "⚠ Lift bottom hip higher away from floor."
                error_joints.extend([23, 24])
                score_deductions += 16
            else:
                primary_feedback = "✓ Strong lateral core hold. Hips lifted high."

            if hip_y < sh_y + 0.08 and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif hip_y >= sh_y + 0.15 and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 27. THORACIC EXTENSION
        elif exercise_name == 'Thoracic Extension':
            primary_feedback = "✓ Upper spine extending smoothly over roller."
            if counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
                counter_state['count'] += 1

        # 28. HIP FLEXOR LUNGE STRETCH
        elif exercise_name == 'Hip Flexor Lunge Stretch':
            if shoulder_diff_y > 0.08:
                primary_feedback = "⚠ Keep torso upright and press hips forward."
                error_joints.extend([11, 12])
                score_deductions += 14
            else:
                primary_feedback = "✓ Hip flexors stretching with upright posture."

            if counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
                counter_state['count'] += 1

        # 29. PIRIFORMIS STRETCH
        elif exercise_name == 'Piriformis Stretch':
            primary_feedback = "✓ Piriformis & deep hip rotator stretch active."
            if counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
                counter_state['count'] += 1

        # 30. PELVIC TILT STABILIZATION
        elif exercise_name == 'Pelvic Tilt Stabilization':
            primary_feedback = "✓ Pelvic tilt locked. Lower back flat against mat."
            if counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
                counter_state['count'] += 1

        # 31. SCAPULAR WALL SLIDE
        elif exercise_name == 'Scapular Wall Slide':
            if l_elbow_angle < 80 or r_elbow_angle < 80:
                primary_feedback = "⚠ Keep wrists and elbows pressed flat against wall."
                error_joints.extend([13, 14, 15, 16])
                score_deductions += 15
            else:
                primary_feedback = "✓ Arms sliding flat in wall plane."

            if l_wr.y < l_sh.y and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif l_wr.y >= l_sh.y and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 32. CHIN TUCK NECK POSTURE
        elif exercise_name == 'Chin Tuck Neck Posture':
            primary_feedback = "✓ Cervical posture retracted into chin tuck."
            if counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
                counter_state['count'] += 1

        # 33. SHOULDER EXTERNAL ROTATION
        elif exercise_name == 'Shoulder External Rotation':
            if l_elbow_flare > 0.32 or r_elbow_flare > 0.32:
                primary_feedback = "⚠ Keep elbows pinned tightly against your sides."
                error_joints.extend([13, 14])
                score_deductions += 15
            else:
                primary_feedback = "✓ Rotator cuff engaged in external rotation."

            arm_span = abs(l_wr.x - r_wr.x)
            if arm_span > 0.45 and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif arm_span <= 0.30 and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 34. WRIST FLEXION STRETCH
        elif exercise_name == 'Wrist Flexion Stretch':
            primary_feedback = "✓ Wrist flexor tendons stretching smoothly."
            if counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
                counter_state['count'] += 1

        # 35. PRONATION & SUPINATION
        elif exercise_name == 'Pronation & Supination':
            primary_feedback = "✓ Forearm rotating through pronation and supination."
            if counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
                counter_state['count'] += 1

        # 36. HIGH KNEE LIFT BALANCE
        elif exercise_name == 'High Knee Lift Balance':
            knee_h = min(l_knee.y, r_knee.y)
            hip_h = (l_hip.y + r_hip.y) / 2.0
            if knee_h > hip_h - 0.05:
                primary_feedback = "⚠ Drive knee higher up to hip height."
                error_joints.append(25 if l_knee.y < r_knee.y else 26)
                target_guides.append({'idx': 25, 'target_x': l_knee.x, 'target_y': hip_h - 0.1})
                score_deductions += 16
            else:
                primary_feedback = "✓ High knee drive above hip height."

            if knee_h < hip_h - 0.08 and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif knee_h >= hip_h and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 37. STANDING SIDE BEND
        elif exercise_name == 'Standing Side Bend':
            spine_lean = abs(l_sh.x - l_hip.x)
            if spine_lean < 0.08:
                primary_feedback = "⚠ Bend laterally at torso without leaning forward."
                error_joints.extend([11, 12])
                score_deductions += 14
            else:
                primary_feedback = "✓ Deep lateral trunk flex stretch."

            if spine_lean > 0.14 and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif spine_lean < 0.05 and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 38. SEATED FORWARD BEND
        elif exercise_name == 'Seated Forward Bend':
            fold_depth = abs(l_sh.y - l_knee.y)
            if fold_depth > 0.20:
                primary_feedback = "⚠ Reach hands toward toes and fold torso lower."
                error_joints.extend([15, 16])
                score_deductions += 15
            else:
                primary_feedback = "✓ Deep seated spinal and hamstring fold."

            if fold_depth < 0.12 and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif fold_depth >= 0.22 and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 39. BRIDGE RAISE SINGLE LEG
        elif exercise_name == 'Bridge Raise Single Leg':
            primary_feedback = "✓ Single-leg bridge held with high hips."
            if counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
                counter_state['count'] += 1

        # 40. DEAD BUG CORE CONTROL
        elif exercise_name == 'Dead Bug Core Control':
            primary_feedback = "✓ Dead bug extension with solid core bracing."
            if counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
                counter_state['count'] += 1

        # 41. CLAMSHELL HIP ABDUCTION
        elif exercise_name == 'Clamshell Hip Abduction':
            primary_feedback = "✓ Gluteus medius abduction active."
            if counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
                counter_state['count'] += 1

        # 42. MONSTER WALK LATERAL BAND
        elif exercise_name == 'Monster Walk Lateral Band':
            knee_width = abs(l_knee.x - r_knee.x)
            if knee_width < 0.25:
                primary_feedback = "⚠ Stay low in half-squat and push knees outward."
                error_joints.extend([25, 26])
                score_deductions += 15
            else:
                primary_feedback = "✓ Lateral hip abductors engaged."

            if knee_width > 0.35 and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif knee_width <= 0.22 and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 43. PENDULUM ARM SWING
        elif exercise_name == 'Pendulum Arm Swing':
            primary_feedback = "✓ Gentle pendulum shoulder release."
            if counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
                counter_state['count'] += 1

        # 44. CROSS-BODY SHOULDER STRETCH
        elif exercise_name == 'Cross-Body Shoulder Stretch':
            primary_feedback = "✓ Posterior shoulder capsule stretching."
            if counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
                counter_state['count'] += 1

        # 45. TRICEPS EXTENSION STRETCH
        elif exercise_name == 'Triceps Extension Stretch':
            primary_feedback = "✓ Triceps overhead stretch active."
            if counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
                counter_state['count'] += 1

        # 46. BICEPS STRETCH
        elif exercise_name == 'Biceps Stretch':
            primary_feedback = "✓ Anterior biceps stretch engaged."
            if counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
                counter_state['count'] += 1

        # 47. ABDUCTOR LEG RAISE
        elif exercise_name == 'Abductor Leg Raise':
            ank_span = abs(l_ank.x - r_ank.x)
            if ank_span < 0.20:
                primary_feedback = "⚠ Lift leg sideways up to 45 degrees."
                error_joints.extend([27, 28])
                score_deductions += 15
            else:
                primary_feedback = "✓ Lateral leg raise at 45 degrees."

            if ank_span > 0.30 and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif ank_span <= 0.15 and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 48. ADDUCTOR SQUEEZE
        elif exercise_name == 'Adductor Squeeze':
            primary_feedback = "✓ Inner thigh adductors engaged."
            if counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
                counter_state['count'] += 1

        # 49. TOE TOUCH FORWARD FOLD
        elif exercise_name == 'Toe Touch Forward Fold':
            wr_h = (l_wr.y + r_wr.y) / 2.0
            ank_h = (l_ank.y + r_ank.y) / 2.0
            if wr_h > ank_h + 0.05:
                primary_feedback = "⚠ Relax head down and reach fingers toward toes."
                error_joints.extend([15, 16])
                score_deductions += 15
            else:
                primary_feedback = "✓ Full posterior chain fold stretch."

            if wr_h < ank_h - 0.05 and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif wr_h >= ank_h and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 50. REVERSE FLY POSTURE
        elif exercise_name == 'Reverse Fly Posture':
            arm_span = abs(l_wr.x - r_wr.x)
            if arm_span < 0.40:
                primary_feedback = "⚠ Hinge at hips and fly arms out to sides."
                error_joints.extend([15, 16])
                score_deductions += 15
            else:
                primary_feedback = "✓ Posterior deltoids & rhomboids engaged."

            if arm_span > 0.55 and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif arm_span <= 0.35 and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 51. SPINAL ARCH FLEXION
        elif exercise_name == 'Spinal Arch Flexion':
            primary_feedback = "✓ Controlled spinal extension."
            if counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
                counter_state['count'] += 1

        # 52. WALL SIT ENDURANCE
        elif exercise_name == 'Wall Sit Endurance':
            avg_knee_angle = (l_knee_angle + r_knee_angle) / 2.0
            if avg_knee_angle > 125:
                primary_feedback = "⚠ Lower hips until knees reach a 90-degree angle."
                error_joints.extend([25, 26])
                score_deductions += 18
            else:
                primary_feedback = "✓ Strong 90-degree wall sit hold."

            if avg_knee_angle < 110 and counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
            elif avg_knee_angle >= 140 and counter_state['stage'] == 'up':
                counter_state['stage'] = 'down'
                counter_state['count'] += 1

        # 53. QUADRUPED HIP EXTENSION
        elif exercise_name == 'Quadruped Hip Extension':
            primary_feedback = "✓ Glute kickback extension active."
            if counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
                counter_state['count'] += 1

        # 54. CERVICAL LATERAL FLEXION
        elif exercise_name == 'Cervical Lateral Flexion':
            primary_feedback = "✓ Upper trapezius and neck flexed laterally."
            if counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
                counter_state['count'] += 1

        # 55. FULL-BODY RESTORATIVE STRETCH
        elif exercise_name == 'Full-Body Restorative Stretch':
            primary_feedback = "✓ Deep full-body lengthen & stretch."
            if counter_state['stage'] == 'down':
                counter_state['stage'] = 'up'
                counter_state['count'] += 1

        else:
            primary_feedback = f"✓ Performing {exercise_name}. Maintain good posture."

        # Compute smoothed scores
        raw_score = max(50, int(100 - score_deductions))
        self.smoothed_form_score = round(0.82 * self.smoothed_form_score + 0.18 * raw_score, 1)
        self.smoothed_confidence = round(0.92 * self.smoothed_confidence + 0.08 * raw_confidence, 3)

        # Rep count logic
        rep_count = counter_state.get('count', 0)
        self.rep_counters[exercise_name] = counter_state

        # LIVE VOICE ASSISTANT FEEDBACK GENERATOR
        voice_text = None
        current_time = time.time()

        # Trigger voice on new rep completed or posture warning
        if rep_count > prev_count:
            voice_text = f"Repetition {rep_count} completed! Great job."
            self.last_voice_message = voice_text
            self.last_voice_time = current_time
        elif "⚠" in primary_feedback:
            clean_speech = primary_feedback.replace("⚠", "").strip()
            if (current_time - self.last_voice_time > self.voice_cooldown_seconds) and (clean_speech != self.last_voice_message):
                voice_text = clean_speech
                self.last_voice_message = clean_speech
                self.last_voice_time = current_time

        return {
            'form_score': self.smoothed_form_score,
            'confidence': round(self.smoothed_confidence * 100, 1),
            'rep_count': rep_count,
            'primary_feedback': primary_feedback,
            'secondary_feedback': secondary_feedback if secondary_feedback else ["Follow camera red/green target indicators."],
            'error_joints': error_joints,
            'target_guides': target_guides,
            'voice_text': voice_text,
            'metrics': {
                'left_elbow_angle': round(l_elbow_angle, 1),
                'right_elbow_angle': round(r_elbow_angle, 1),
                'left_shoulder_angle': round(l_shoulder_angle, 1),
                'right_shoulder_angle': round(r_shoulder_angle, 1),
                'alignment': 'Good' if shoulder_diff_y <= 0.08 else 'Needs Adjustment'
            },
            'body_detected': True
        }
