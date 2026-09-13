import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

def generate_synthetic_dataset(num_samples_per_class=150, num_coords=501):
    """Generates a realistic synthetic posture landmark dataset for default exercises if CSV is missing."""
    classes = ['Shoulder Rotation', 'Wrist Extension', 'Spinal Twist', 'Arm Rotation']
    data = []
    
    for cls in classes:
        # Create distinct landmark signature patterns for each exercise
        base_seed = hash(cls) % 1000
        np.random.seed(base_seed)
        
        for _ in range(num_samples_per_class):
            row = [cls]
            # 501 landmarks * 4 (x, y, z, v) = 2004 features
            if cls == 'Shoulder Rotation':
                feature_vec = np.random.normal(loc=0.5, scale=0.08, size=num_coords * 4)
                feature_vec[10:30] += 0.25 # Shoulder joint signature
            elif cls == 'Wrist Extension':
                feature_vec = np.random.normal(loc=0.3, scale=0.08, size=num_coords * 4)
                feature_vec[30:50] += 0.35 # Wrist joint signature
            elif cls == 'Spinal Twist':
                feature_vec = np.random.normal(loc=0.7, scale=0.08, size=num_coords * 4)
                feature_vec[50:80] += 0.45 # Spine signature
            else: # Arm Rotation
                feature_vec = np.random.normal(loc=0.1, scale=0.08, size=num_coords * 4)
                feature_vec[0:20] += 0.50 # Arm signature
                
            row.extend(feature_vec)
            data.append(row)
            
    num_coords_cols = num_coords
    cols = ['class']
    for val in range(1, num_coords_cols + 1):
        cols.extend([f'x{val}', f'y{val}', f'z{val}', f'v{val}'])
        
    df = pd.DataFrame(data, columns=cols)
    os.makedirs('dataset', exist_ok=True)
    df.to_csv('dataset/coords1.csv', index=False)
    print(f"[+] Created initial benchmark dataset at 'dataset/coords1.csv' with {len(df)} samples.")
    return df

def train_model():
    print("=" * 60)
    print(" AI Exercise Classifier - Training Pipeline")
    print("=" * 60)

    csv_path = 'dataset/coords1.csv'
    if not os.path.exists(csv_path):
        print("[!] No existing dataset found. Generating initial dataset...")
        df = generate_synthetic_dataset()
    else:
        df = pd.read_csv(csv_path)

    print(f"[+] Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    X = df.drop('class', axis=1)
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234, stratify=y)

    pipelines = {
        'lr': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
        'rf': make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42)),
        'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=42))
    }

    fit_models = {}
    best_model_name = None
    best_accuracy = 0.0

    print("\n--- Training Benchmark ---")
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        fit_models[name] = pipeline
        
        yhat = pipeline.predict(X_test)
        acc = accuracy_score(y_test, yhat)
        print(f"Model [{name.upper()}] Accuracy: {acc * 100:.2f}%")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name

    print(f"\n[+] Best Model: {best_model_name.upper()} ({best_accuracy * 100:.2f}% Accuracy)")

    best_pipeline = fit_models[best_model_name]

    model_path = 'body_language.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_pipeline, f)
        
    print(f"[+] Model successfully saved to '{model_path}'.")

if __name__ == '__main__':
    train_model()
