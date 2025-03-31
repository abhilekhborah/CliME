import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import matplotlib.patheffects as PathEffects

MODEL_NAME = "Model Name"

def load_climate_data(file_path):
    """
    Load and process climate messaging data.
    Returns a DataFrame with CAQ dimensions.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading JSON file: {e}")
        print("Attempting alternative parsing method...")
        try:
            with open(file_path, 'r') as file:
                text = file.read()
                start_idx = text.find('{')
                end_idx = text.rfind('}') + 1
                if start_idx >= 0 and end_idx > 0:
                    data = json.loads(text[start_idx:end_idx])
                else:
                    print("Could not identify valid JSON structure in file")
                    return pd.DataFrame()
        except Exception as e2:
            print(f"Secondary parsing attempt failed: {e2}")
            return pd.DataFrame()
    
    results = []
    for msg_id, msg_data in data.items():
        if not isinstance(msg_data, dict):
            continue
        msg_text = msg_data.get('text', f"Message {msg_id}")
        try:
            actionability_caq = float(msg_data.get('actionability', {}).get('caq', np.nan))
            criticality_caq = float(msg_data.get('criticality', {}).get('caq', np.nan))
            justice_caq = float(msg_data.get('justice', {}).get('caq', np.nan))
            results.append({
                'message_id': msg_id,
                'message_text': msg_text[:50] + '...' if len(msg_text) > 50 else msg_text,
                'actionability_caq': actionability_caq,
                'criticality_caq': criticality_caq,
                'justice_caq': justice_caq
            })
        except (AttributeError, ValueError, TypeError) as e:
            print(f"Warning: Could not extract CAQ scores for message {msg_id}: {e}")
    
    df = pd.DataFrame(results)
    before_count = len(df)
    df = df.dropna(subset=['actionability_caq', 'criticality_caq', 'justice_caq'])
    after_count = len(df)
    if before_count > after_count:
        print(f"Note: Removed {before_count - after_count} messages with missing data")
    return df

def create_enhanced_radar_plot(df, figsize=(12, 10), 
                               save_path=f'{MODEL_NAME.replace(" ", "_")}_enhanced_caq_radar.png'):
    """
    Create a radar plot showing mean CAQ values and distribution metrics.
    """
    # Calculate statistics
    stats = {
        'mean': df[['actionability_caq', 'criticality_caq', 'justice_caq']].mean(),
        'std': df[['actionability_caq', 'criticality_caq', 'justice_caq']].std(),
        'min': df[['actionability_caq', 'criticality_caq', 'justice_caq']].min(),
        'max': df[['actionability_caq', 'criticality_caq', 'justice_caq']].max(),
        'q25': df[['actionability_caq', 'criticality_caq', 'justice_caq']].quantile(0.25),
        'q75': df[['actionability_caq', 'criticality_caq', 'justice_caq']].quantile(0.75),
        'median': df[['actionability_caq', 'criticality_caq', 'justice_caq']].median()
    }
    
    # Prepare values for radar plot (closing the polygon)
    mean_values = [
        stats['mean']['actionability_caq'],
        stats['mean']['criticality_caq'],
        stats['mean']['justice_caq'],
        stats['mean']['actionability_caq']
    ]
    std_values = [
        stats['std']['actionability_caq'],
        stats['std']['criticality_caq'],
        stats['std']['justice_caq'],
        stats['std']['actionability_caq']
    ]
    min_values = [
        stats['min']['actionability_caq'],
        stats['min']['criticality_caq'],
        stats['min']['justice_caq'],
        stats['min']['actionability_caq']
    ]
    max_values = [
        stats['max']['actionability_caq'],
        stats['max']['criticality_caq'],
        stats['max']['justice_caq'],
        stats['max']['actionability_caq']
    ]
    q25_values = [
        stats['q25']['actionability_caq'],
        stats['q25']['criticality_caq'],
        stats['q25']['justice_caq'],
        stats['q25']['actionability_caq']
    ]
    q75_values = [
        stats['q75']['actionability_caq'],
        stats['q75']['criticality_caq'],
        stats['q75']['justice_caq'],
        stats['q75']['actionability_caq']
    ]
    median_values = [
        stats['median']['actionability_caq'],
        stats['median']['criticality_caq'],
        stats['median']['justice_caq'],
        stats['median']['actionability_caq']
    ]
    
    categories = ['Actionability', 'Criticality', 'Justice', 'Actionability']
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=True)
    
    main_color = '#3498db'
    accent_color = '#e74c3c'
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=figsize, facecolor='#f8f8f8')
    ax = fig.add_subplot(111, polar=True, facecolor='#f9f9f9')
    
    # Draw concentric circles with labels
    circle_levels = [0.2, 0.4, 0.6, 0.8]
    for level in circle_levels:
        ax.plot(np.linspace(0, 2*np.pi, 100), [level] * 100, 
                linestyle='--', color='gray', alpha=0.3, linewidth=0.8)
        text = ax.text(np.pi/4, level, f'{level:.1f}', 
                       horizontalalignment='center', verticalalignment='center', 
                       fontsize=9, color='gray', alpha=0.8)
        text.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='white', alpha=0.8)])
    
    # Plot the mean CAQ polygon with a slight border
    ax.fill(angles, mean_values, alpha=0.25, color=main_color, label='Mean CAQ')
    ax.plot(angles, mean_values, '-', linewidth=2, color=main_color)
    
    # Add dimension labels
    max_value = max(max_values[:-1]) * 1.15
    for angle, category in zip(angles[:-1], categories[:-1]):
        ha = 'center'
        if angle < np.pi/2 or angle > 3*np.pi/2:
            ha = 'left'
        elif np.pi/2 < angle < 3*np.pi/2:
            ha = 'right'
        text = ax.text(angle, max_value, category, size=16, fontweight='bold', ha=ha, va='center')
        text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white', alpha=0.8)])
    
    # Annotate mean values near the data points
    for angle, value in zip(angles[:-1], mean_values[:-1]):
        text_distance = value + 0.08
        text = ax.text(angle, text_distance, f"{value:.3f}", 
                       horizontalalignment='center', verticalalignment='center',
                       fontsize=13, fontweight='bold', color=main_color)
        text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white', alpha=0.9)])
    
    ax.set_ylim(0, max(max_values) * 1.2)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    
    plt.title(f'{MODEL_NAME}: CAQ Dimensions Analysis', 
              fontsize=22, fontweight='bold', pad=30, color='#2c3e50')
    
    legend = ax.legend(loc='upper right', fontsize=12, frameon=True, 
                       framealpha=0.9, edgecolor='#cccccc', bbox_to_anchor=(0.1, 0.1))
    legend.get_frame().set_facecolor('#ffffff')
    
    stats_text = (
        f"{MODEL_NAME} STATISTICS\n"
        f"───────────────────────\n"
        f"CAQ (Actionability): {mean_values[0]:.3f} ± {std_values[0]:.3f}\n"
        f"CAQ (Criticality): {mean_values[1]:.3f} ± {std_values[1]:.3f}\n"
        f"CAQ (Justice): {mean_values[2]:.3f} ± {std_values[2]:.3f}\n\n"
        f"Mean: {np.mean(mean_values[:-1]):.3f}\n"
        f"Descriptors Analyzed: {len(df)}"
    )
    plt.figtext(0.7, 0.05, stats_text, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.8', edgecolor='#cccccc'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Radar plot saved as '{save_path}'")
    
    return fig

if __name__ == "__main__":
    file_path = "/Users/deltae/Downloads/AIISC/ACL - NLPforPosImp/revised_qwen_results.json"
    dir = "/scratch/user/hasnat.md.abdullah/Climmeme/outputs/revised_caq"
    
    # Traverse all JSON files in the directory and generate plots
    for root, _, files in os.walk(dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                df = load_climate_data(file_path)
                
                if df.empty:
                    print(f"Warning: No valid data found in {file_path}. Skipping...")
                else:
                    print(f"Successfully loaded {len(df)} messages with CAQ dimensions from {file_path}.")
                    save_path = os.path.join(root, f"{os.path.splitext(file)[0]}_enhanced_caq_radar.png")
                    print(f"Creating enhanced radar plot for {file_path}...")
                    create_enhanced_radar_plot(df, save_path=save_path)
                    print(f"Plot saved to {save_path}.")
    
