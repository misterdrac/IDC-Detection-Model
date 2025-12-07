import os
import glob
import pandas as pd
# Import the necessary grouping splitter
from sklearn.model_selection import StratifiedGroupKFold

# CONFIGURATION
# Update this path to where you extracted the dataset
# Dataset structure expected: root_dir / patient_id / class (0 or 1) / image.png
DATASET_PATH = 'D:/.Programming/datasets/IDC/data/IDC_regular_ps50_idx5'


def create_splits_by_patient(dataset_path, n_splits=5):
    print(f"Scanning files in {dataset_path}...")

    # 1. Gather all file paths
    # The glob pattern looks for: root -> any folder -> 0 or 1 -> *.png
    file_pattern = os.path.join(dataset_path, '**', '*.png')
    all_image_paths = glob.glob(file_pattern, recursive=True)

    if len(all_image_paths) == 0:
        print("No images found! Please check your DATASET_PATH.")
        return

    print(f"Found {len(all_image_paths)} images. Processing metadata...")

    # 2. Extract metadata (Patient ID and Target) from paths
    data = []
    for filepath in all_image_paths:
        parts = filepath.split(os.sep)
        filename = parts[-1]

        if 'class0.png' in filename:
            label = 0
        elif 'class1.png' in filename:
            label = 1
        else:
            continue  # Skip files that don't match pattern

        # Assuming structure is /IDC_regular_ps50_idx5/patient_id/class/img.png
        # The patient ID is the folder name two levels up from the image file.
        patient_id = parts[-3]

        data.append({
            'path': filepath,
            'patient_id': patient_id,
            'target': label
        })

    df = pd.DataFrame(data)
    print(f"Dataframe created with {len(df)} entries from {df['patient_id'].nunique()} unique patients.")

    # --- STEP 3: Create Stratified Group K-Folds (The Key Change) ---

    # Use StratifiedGroupKFold to ensure:
    # 1. Class balance (Stratified)
    # 2. No patient is in both train and validation folds (Group)
    df['fold'] = -1
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # We split using three arguments: X (data to split), y (target for stratification),
    # and groups (patient_id for grouping).
    print("Generating Stratified Group Folds...")
    for fold_num, (train_idx, val_idx) in enumerate(sgkf.split(
            X=df['path'],
            y=df['target'],
            groups=df['patient_id']
    )):
        df.loc[val_idx, 'fold'] = fold_num

    # --- Verification ---
    # Check for leakage (optional but recommended)
    for i in range(n_splits):
        val_patients = set(df[df['fold'] == i]['patient_id'].unique())
        train_patients = set(df[df['fold'] != i]['patient_id'].unique())

        # Intersection should be empty
        leakage = val_patients.intersection(train_patients)
        if leakage:
            print(f"🚨 WARNING: Fold {i} has patient leakage! {len(leakage)} shared patients.")

    # 4. Save to CSV
    output_file = 'breast_cancer_5fold_patient_splits.csv'
    df.to_csv(output_file, index=False)

    print(f"\n✅ Success! Patient-split Folds saved to '{output_file}'.")
    print("Class distribution per fold:")
    for i in range(n_splits):
        fold_data = df[df['fold'] == i]
        ratio = fold_data['target'].mean()
        print(
            f"Fold {i}: {len(fold_data)} images ({fold_data['patient_id'].nunique()} patients), IDC+ Ratio: {ratio:.2%}")


if __name__ == "__main__":
    create_splits_by_patient(DATASET_PATH)