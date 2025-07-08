import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# --- Config ---
BASELINE_LOG_PATTERN = r'baseline_checkpoints\baseline_loss_log_20250708_105102.csv'
MAML_LOG_PATTERN = r'maml_checkpoints\maml_log_20250708_141656.csv'

# --- Helper Functions ---
def load_baseline_log(filepath):
    df = pd.read_csv(filepath)
    # Only keep rows with batch == '' for epoch summary
    epoch_df = df[df['Batch'].isnull()].copy()
    epoch_df['Epoch'] = epoch_df['Epoch'].astype(int)
    return epoch_df[['Epoch', 'TrainLoss', 'Val_mAP']]

def load_maml_log(filepath):
    df = pd.read_csv(filepath)
    df['Epoch'] = df['Epoch'].astype(int)
    return df[['Epoch', 'SupportLoss', 'QueryLoss', 'Val_mAP']]

# --- Load Logs ---
baseline_logs = sorted(glob.glob(BASELINE_LOG_PATTERN))
maml_logs = sorted(glob.glob(MAML_LOG_PATTERN))

if not baseline_logs:
    raise FileNotFoundError('No baseline log files found!')
if not maml_logs:
    raise FileNotFoundError('No MAML log files found!')

baseline_df = load_baseline_log(baseline_logs[-1])  # Use latest
maml_df = load_maml_log(maml_logs[-1])  # Use latest

# --- Plotting ---
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Baseline Training Loss
axs[0, 0].plot(baseline_df['Epoch'], baseline_df['TrainLoss'], label='Baseline Train Loss', color='tab:blue')
axs[0, 0].set_title('Baseline Training Loss')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].legend()

# MAML Support/Query Loss
axs[0, 1].plot(maml_df['Epoch'], maml_df['SupportLoss'], label='MAML Support Loss', color='tab:orange')
axs[0, 1].plot(maml_df['Epoch'], maml_df['QueryLoss'], label='MAML Query Loss', color='tab:green')
axs[0, 1].set_title('MAML Support & Query Loss')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Loss')
axs[0, 1].legend()

# Baseline Val mAP
axs[1, 0].plot(baseline_df['Epoch'], baseline_df['Val_mAP'], label='Baseline Val mAP', color='tab:blue')
axs[1, 0].set_title('Baseline Validation mAP')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('mAP')
axs[1, 0].legend()

# MAML Val mAP
axs[1, 1].plot(maml_df['Epoch'], maml_df['Val_mAP'], label='MAML Val mAP', color='tab:orange')
axs[1, 1].set_title('MAML Validation mAP')
axs[1, 1].set_xlabel('Epoch')
axs[1, 1].set_ylabel('mAP')
axs[1, 1].legend()

plt.tight_layout()
plt.show()

# --- Summary Statistics ---
def print_summary(df, name, loss_cols, map_col):
    print(f'--- {name} ---')
    for col in loss_cols:
        print(f'  Final {col}: {df[col].iloc[-1]:.4f}')
    print(f'  Best Val mAP: {df[map_col].max():.4f} (Epoch {df[map_col].idxmax()})')
    print()

print_summary(baseline_df, 'Baseline', ['TrainLoss'], 'Val_mAP')
print_summary(maml_df, 'MAML', ['SupportLoss', 'QueryLoss'], 'Val_mAP') 