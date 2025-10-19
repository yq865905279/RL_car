"""
Training Results Visualization Script
Plot comparison and individual training curves for different models
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d

# Set font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Data directory
DATA_DIR = Path("training_result")
OUTPUT_DIR = Path("training_result/plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# Metric file mapping
METRICS = {
    "avg_reward_1000": {"file": "avg_reward_1000.csv", "ylabel": "Average Reward (1000 steps)", "title": "Average Reward"},
    "success_rate_recent_100": {"file": "success_rate_recent_100.csv", "ylabel": "Success Rate", "title": "Success Rate"},
    "avg_linear_vel_window": {"file": "avg_linear_vel_window.csv", "ylabel": "Linear Velocity (m/s)", "title": "Average Linear Velocity"},
    "avg_linear_acc_window": {"file": "avg_linear_acc_window.csv", "ylabel": "Linear Acceleration (m/s²)", "title": "Average Linear Acceleration"}
}

def clean_value(val):
    """Clean data values (remove quotes, etc.)"""
    if isinstance(val, str):
        return float(val.strip("'\""))
    return float(val)

def load_data(metric_key):
    """Load metric data"""
    file_path = DATA_DIR / METRICS[metric_key]["file"]
    df = pd.read_csv(file_path)
    
    # Clean quotes in Run column
    df['Run'] = df['Run'].str.strip('"')
    
    # Clean value column
    df['value'] = df['value'].apply(clean_value)
    
    return df

def get_short_run_name(run_name):
    """Generate short Run name for legend"""
    # Extract key information: cargo type and version
    parts = run_name.split('_')
    cargo_type = None
    version = None
    
    for part in parts:
        if part in ['normal', 'fragile', 'dangerous']:
            cargo_type = part
        if part.startswith('T'):
            version = part
    
    if cargo_type and version:
        return f"{version}_{cargo_type}"
    return run_name

def smooth_curve(values, window_size=20):
    """Smooth curve using uniform filter"""
    if len(values) < window_size:
        window_size = max(3, len(values) // 5)
    return uniform_filter1d(values, size=window_size)

def plot_comparison_all_metrics():
    """Plot comparison of all metrics (2x2 subplots)"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Results Comparison - All Models', fontsize=16, fontweight='bold')
    
    axes_flat = axes.flatten()
    
    # Define color palette for models
    colors = plt.cm.tab10(np.linspace(0, 1, 10))  # Get 10 distinct colors
    
    for idx, (metric_key, metric_info) in enumerate(METRICS.items()):
        ax = axes_flat[idx]
        df = load_data(metric_key)
        
        # Get all unique Runs
        runs = df['Run'].unique()
        
        # Plot curve for each Run with unique color
        for run_idx, run in enumerate(runs):
            run_data = df[df['Run'] == run].sort_values('step')
            short_name = get_short_run_name(run)
            color = colors[run_idx % len(colors)]  # Assign unique color
            
            # Plot raw data with transparency
            ax.plot(run_data['step'], run_data['value'], 
                   color=color, linewidth=1.5, alpha=0.25)
            
            # Plot smoothed curve with same color
            smoothed = smooth_curve(run_data['value'].values)
            ax.plot(run_data['step'], smoothed,
                   color=color, label=short_name, linewidth=2.5, alpha=0.9)
        
        ax.set_xlabel('Training Steps', fontsize=11)
        ax.set_ylabel(metric_info['ylabel'], fontsize=11)
        ax.set_title(metric_info['title'], fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "comparison_all_metrics.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {save_path}")
    plt.close()

def plot_comparison_single_metric(metric_key):
    """Plot comparison of all models for a single metric"""
    metric_info = METRICS[metric_key]
    df = load_data(metric_key)
    
    plt.figure(figsize=(12, 7))
    
    runs = df['Run'].unique()
    
    # Define color palette for models
    colors = plt.cm.tab10(np.linspace(0, 1, 10))  # Get 10 distinct colors
    
    for run_idx, run in enumerate(runs):
        run_data = df[df['Run'] == run].sort_values('step')
        short_name = get_short_run_name(run)
        color = colors[run_idx % len(colors)]  # Assign unique color
        
        # Plot raw data with transparency
        plt.plot(run_data['step'], run_data['value'], 
                color=color, linewidth=1.5, alpha=0.25)
        
        # Plot smoothed curve with same color
        smoothed = smooth_curve(run_data['value'].values)
        plt.plot(run_data['step'], smoothed,
                color=color, label=short_name, linewidth=2.5, alpha=0.9)
    
    plt.xlabel('Training Steps', fontsize=13)
    plt.ylabel(metric_info['ylabel'], fontsize=13)
    plt.title(f'{metric_info["title"]} - All Models Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    save_path = OUTPUT_DIR / f"comparison_{metric_key}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {save_path}")
    plt.close()

def plot_single_run(run_name):
    """Plot all metrics for a single Run (2x2 subplots)"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    short_name = get_short_run_name(run_name)
    fig.suptitle(f'Training Results - {short_name}', fontsize=16, fontweight='bold')
    
    axes_flat = axes.flatten()
    
    for idx, (metric_key, metric_info) in enumerate(METRICS.items()):
        ax = axes_flat[idx]
        df = load_data(metric_key)
        
        # Filter data for current Run
        run_data = df[df['Run'] == run_name].sort_values('step')
        
        if len(run_data) > 0:
            # Plot raw data
            ax.plot(run_data['step'], run_data['value'], 
                   linewidth=1.5, color='#2E86AB', alpha=0.4, label='Raw')
            
            # Add smoothed curve
            if len(run_data) > 10:
                smoothed = smooth_curve(run_data['value'].values)
                ax.plot(run_data['step'], smoothed, 
                       linewidth=3, color='#A23B72', alpha=0.8, 
                       label='Smoothed')
            
            ax.set_xlabel('Training Steps', fontsize=11)
            ax.set_ylabel(metric_info['ylabel'], fontsize=11)
            ax.set_title(metric_info['title'], fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            if len(run_data) > 10:
                ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
    
    plt.tight_layout()
    
    # Create safe filename
    safe_name = run_name.replace('"', '').replace('/', '_')
    save_path = OUTPUT_DIR / f"single_run_{safe_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved single run plot: {save_path}")
    plt.close()

def generate_statistics_table():
    """Generate statistics table"""
    print("\n" + "="*80)
    print("Training Results Statistics Summary")
    print("="*80)
    
    # Load all data
    all_data = {}
    for metric_key in METRICS.keys():
        all_data[metric_key] = load_data(metric_key)
    
    # Get all Runs
    runs = all_data['avg_reward_1000']['Run'].unique()
    
    stats_data = []
    for run in runs:
        short_name = get_short_run_name(run)
        stats = {'Model': short_name}
        
        for metric_key, metric_info in METRICS.items():
            df = all_data[metric_key]
            run_data = df[df['Run'] == run]
            
            if len(run_data) > 0:
                values = run_data['value'].values
                stats[f'{metric_info["title"]}_Final'] = f"{values[-1]:.4f}"
                stats[f'{metric_info["title"]}_Max'] = f"{np.max(values):.4f}"
                stats[f'{metric_info["title"]}_Mean'] = f"{np.mean(values):.4f}"
        
        stats_data.append(stats)
    
    # Print statistics table
    df_stats = pd.DataFrame(stats_data)
    print(df_stats.to_string(index=False))
    print("="*80 + "\n")
    
    # Save to CSV
    csv_path = OUTPUT_DIR / "training_statistics.csv"
    df_stats.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ Saved statistics table: {csv_path}\n")

def main():
    """Main function"""
    print("\n" + "="*80)
    print("Starting training results visualization...")
    print("="*80 + "\n")
    
    # 1. Generate comparison plot for all metrics (2x2)
    print("1. Generating all metrics comparison plot...")
    plot_comparison_all_metrics()
    
    # 2. Generate individual comparison plots for each metric
    print("\n2. Generating individual metric comparison plots...")
    for metric_key in METRICS.keys():
        plot_comparison_single_metric(metric_key)
    
    # 3. Generate individual plots for each Run
    print("\n3. Generating individual model plots...")
    df_sample = load_data('avg_reward_1000')
    runs = df_sample['Run'].unique()
    
    for run in runs:
        plot_single_run(run)
    
    # 4. Generate statistics table
    print("\n4. Generating statistics table...")
    generate_statistics_table()
    
    print("="*80)
    print(f"✅ All plots generated successfully! Output directory: {OUTPUT_DIR}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
