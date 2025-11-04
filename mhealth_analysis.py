# ==========================================
# mHealth Dataset Analysis: Folder Version
# Combines all TXT files in current folder
# ==========================================
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# ---------------- Setup ----------------
RANDOM_STATE = 42
os.makedirs("outputs", exist_ok=True)

# --------- 1. Load and combine files -------------
def load_mhealth_folder(folder="."):
    """Load all .txt or .log files in current folder and subfolders"""
    files = glob.glob(os.path.join(folder, "*.txt")) + \
            glob.glob(os.path.join(folder, "*.log"))
    if not files:
        raise FileNotFoundError(f"No data files found in {folder}.")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, sep=r"\s+", header=None)
            df["__source_file"] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")
    big = pd.concat(dfs, ignore_index=True)
    return big

print("Loading dataset...")
raw = load_mhealth_folder("MHEALTHDATASET")
print("✅ Dataset loaded successfully!")
print("Shape:", raw.shape)

# --------- 2. Assign column names -------------
cols = [
    "acc_chest_x","acc_chest_y","acc_chest_z",
    "ecg_lead1","ecg_lead2",
    "acc_left_x","acc_left_y","acc_left_z",
    "gyro_left_x","gyro_left_y","gyro_left_z",
    "mag_left_x","mag_left_y","mag_left_z",
    "acc_right_x","acc_right_y","acc_right_z",
    "gyro_right_x","gyro_right_y","gyro_right_z",
    "mag_right_x","mag_right_y","mag_right_z",
    "activity_id"
]
if raw.shape[1] >= len(cols):
    raw.columns = cols + ["__source_file"]
else:
    raw.columns = cols[:raw.shape[1]]
print("✅ Columns assigned.")

# --------- 3. Compute vector magnitudes ---------
def compute_vm(df, prefix):
    x, y, z = f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"
    out = f"{prefix}_vm"
    df[out] = np.sqrt(df[x]**2 + df[y]**2 + df[z]**2)
    return out

vm_cols = []
for p in ["acc_chest","acc_left","gyro_left","mag_left","acc_right","gyro_right","mag_right"]:
    if all(f"{p}_{ax}" in raw.columns for ax in ["x","y","z"]):
        vm_cols.append(compute_vm(raw, p))

print("✅ Vector magnitude columns created:", vm_cols)

# --------- 4. Correlation analysis ---------
corr = raw[vm_cols].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation between sensor vector magnitudes")
plt.tight_layout()
plt.savefig("outputs/vm_corr_heatmap.png")
plt.close()
corr.to_csv("outputs/vm_correlations.csv")
print("✅ Saved correlation matrix and heatmap.")

# --------- 5. Activity-wise summary ---------
stats = raw.groupby("activity_id")[vm_cols].agg(["mean","std","median","min","max"])
stats.to_csv("outputs/activity_vm_stats.csv")
print("✅ Saved per-activity statistics.")
# --------- 5.1 Activity distribution and boxplots ---------
# Check if activity_id exists
if "activity_id" in raw.columns:
    # --- Activity distribution ---
    plt.figure(figsize=(10,5))
    raw["activity_id"].value_counts().sort_index().plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Activity Distribution")
    plt.xlabel("Activity ID")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("outputs/activity_distribution.png")
    plt.close()

    # --- Activity boxplots for movement intensity ---
    vm_for_plot = [c for c in raw.columns if c.endswith("_vm")]
    for col in vm_for_plot:
        plt.figure(figsize=(10,5))
        sns.boxplot(x="activity_id", y=col, data=raw)
        plt.title(f"Distribution of {col} across Activities")
        plt.xlabel("Activity ID")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(f"outputs/{col}_boxplot.png")
        plt.close()

    print("✅ Saved activity distribution and VM boxplots.")

# --------- 6. Build transactions for pattern mining ---------
WINDOW_SIZE = 50
THRESH_STD = 1.0
TOP_K = 3

transactions, labels = [], []
num_rows = raw.shape[0]

for start in range(0, num_rows, WINDOW_SIZE):
    chunk = raw.iloc[start:start+WINDOW_SIZE]
    if chunk.empty:
        continue
    active = []
    for col in vm_cols:
        mu, sd = chunk[col].mean(), chunk[col].std()
        thresh = mu + THRESH_STD * sd
        if (chunk[col] > thresh).sum() > (len(chunk)//2):
            active.append(col)
    if not active:
        active = chunk[vm_cols].mean().sort_values(ascending=False).head(TOP_K).index.tolist()
    transactions.append(active)
    labels.append(int(chunk["activity_id"].mode()[0]))

pd.DataFrame({
    "transaction":[";".join(t) for t in transactions],
    "activity":labels
}).to_csv("outputs/transactions.csv", index=False)
print(f"✅ Built {len(transactions)} transactions.")

# --------- 7. Frequent pattern mining ---------
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
trans_df = pd.DataFrame(te_ary, columns=te.columns_)

min_support = 0.05
min_confidence = 0.6

# Apriori
freq_ap = apriori(trans_df, min_support=min_support, use_colnames=True)
freq_ap.to_csv("outputs/frequent_itemsets_apriori.csv", index=False)
rules_ap = association_rules(freq_ap, metric="confidence", min_threshold=min_confidence)
rules_ap.sort_values(by=["confidence","lift"], ascending=False).to_csv("outputs/association_rules_apriori.csv", index=False)

# FP-Growth
freq_fp = fpgrowth(trans_df, min_support=min_support, use_colnames=True)
freq_fp.to_csv("outputs/frequent_itemsets_fpgrowth.csv", index=False)

print("✅ Frequent pattern mining complete.")
print("Apriori itemsets:", freq_ap.shape[0])
print("FP-Growth itemsets:", freq_fp.shape[0])

# --------- 8. Rule visualizations ---------
if not rules_ap.empty:
    plt.figure(figsize=(8,6))
    plt.scatter(rules_ap["confidence"], rules_ap["lift"],
                s=rules_ap["support"]*1000, alpha=0.6, edgecolors="k")
    plt.xlabel("Confidence")
    plt.ylabel("Lift")
    #-------Lift vs confidence-------
    plt.title("Association Rules: Lift vs Confidence")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("outputs/rules_scatter.png")
    plt.close()

    rules_ap["rule"] = rules_ap["antecedents"].astype(str) + " → " + rules_ap["consequents"].astype(str)
    top_rules = rules_ap.nlargest(10, "lift")
    plt.figure(figsize=(10,6))
    plt.barh(top_rules["rule"], top_rules["lift"], color="skyblue")
    plt.xlabel("Lift")
    plt.title("Top 10 Rules by Lift (Apriori)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("outputs/rules_top_lift.png")
    plt.close()
    print("✅ Saved: outputs/rules_scatter.png and outputs/rules_top_lift.png")
else:
    print("⚠️ No rules found to visualize.")

# --------- 9. Save README ---------
with open("outputs/README.txt","w") as f:
    f.write("Generated by mhealth_analysis.py\n")
    f.write("Outputs:\n")
    f.write(" - vm_corr_heatmap.png\n")
    f.write(" - vm_correlations.csv\n")
    f.write(" - activity_vm_stats.csv\n")
    f.write(" - transactions.csv\n")
    f.write(" - frequent_itemsets_apriori.csv\n")
    f.write(" - frequent_itemsets_fpgrowth.csv\n")
    f.write(" - association_rules_apriori.csv\n")
    f.write(" - rules_scatter.png\n")
    f.write(" - rules_top_lift.png\n")

print("\n✅ All analysis complete! Check the 'outputs/' folder for results.")