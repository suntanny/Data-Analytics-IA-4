import os
import pandas as pd

# Path to your dataset folder (relative to this script)
data_dir = "MHEALTHDATASET"

# Column names from README
columns = [
    "timestamp",
    "chest_acc_x", "chest_acc_y", "chest_acc_z",
    "chest_gyro_x", "chest_gyro_y", "chest_gyro_z",
    "chest_mag_x", "chest_mag_y", "chest_mag_z",
    "right_acc_x", "right_acc_y", "right_acc_z",
    "left_acc_x", "left_acc_y", "left_acc_z",
    "left_gyro_x", "left_gyro_y", "left_gyro_z",
    "left_mag_x", "left_mag_y", "left_mag_z",
    "ecg_1", "ecg_2",
    "activity"
]

# Activity label mapping
activity_labels = {
    0: "null",
    1: "standing",
    2: "sitting",
    3: "lying_down",
    4: "walking",
    5: "climbing_stairs",
    6: "waist_bends_forward",
    7: "frontal_elevation_of_arms",
    8: "knees_bending",
    9: "cycling",
    10: "jogging",
    11: "running",
    12: "jump_front_back"
}

# Load all subjects
all_data = []

print("📦 Loading MHEALTH dataset...\n")

for file in sorted(os.listdir(data_dir)):
    if file.endswith(".log"):
        path = os.path.join(data_dir, file)
        print(f"Reading {file} ...")
        df = pd.read_csv(path, sep="\t", header=None, names=columns)
        df["subject"] = file.split("_")[1].replace(".log", "")
        all_data.append(df)

# Combine into one DataFrame
mhealth_df = pd.concat(all_data, ignore_index=True)

# Map activity codes to names
mhealth_df["activity_name"] = mhealth_df["activity"].map(activity_labels)

print("\n✅ Dataset loaded successfully!")
print(f"Shape: {mhealth_df.shape}")
print("\nSample rows:")
print(mhealth_df.head())

# Optional: save the combined dataset
mhealth_df.to_csv("MHEALTH_combined.csv", index=False)
print("\n💾 Saved combined dataset as 'MHEALTH_combined.csv'")
